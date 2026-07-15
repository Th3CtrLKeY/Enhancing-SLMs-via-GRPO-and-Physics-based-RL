"""
Modal deployment for the Marine Hydrodynamics Gradio chat demo.

Requires a free Modal account (~$30/month compute credits on Starter):
    pip install modal
    modal setup
    modal deploy app/modal_app.py

The app scales to zero when idle (no charge). First cold start downloads the
~3B base model into a persistent Volume; later starts reuse the cache.
"""

from __future__ import annotations

import modal

APP_NAME = "marine-hydro-slm-chat"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("marine-hydro-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
        "gradio==5.6.0",
        "fastapi==0.115.5",
        "huggingface_hub==0.26.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "ADAPTER_PATH": "/lora",
            "BASE_MODEL": "Qwen/Qwen2.5-3B-Instruct",
            "MAX_NEW_TOKENS": "512",
            "TEMPERATURE": "0.25",
        }
    )
    .add_local_dir(
        "grpo_run3_output",
        remote_path="/lora",
        # Skip bulky training logs; only adapter weights are needed at inference.
        ignore=["*.jsonl", "README.md"],
    )
    .add_local_python_source("app")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 30,
    scaledown_window=120,
    max_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.concurrent(max_inputs=40)
@modal.asgi_app()
def serve():
    """ASGI Gradio app on a Modal T4 (scales to zero when idle)."""
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    from app.gradio_app import build_demo
    from app.inference import get_model

    get_model()
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")


@app.local_entrypoint()
def main() -> None:
    print("Deploy with:  modal deploy app/modal_app.py")
    print("Or serve once: modal serve app/modal_app.py")
