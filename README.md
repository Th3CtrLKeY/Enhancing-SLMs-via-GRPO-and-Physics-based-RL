# Enhancing SLMs via GRPO and Physics-based RL

Research codebase for supervised fine-tuning (SFT), Group Relative Policy Optimization (GRPO), and physics-grounded reward shaping on small language models.

## Chat demo (working app)

A Gradio Q&A UI serves the fine-tuned marine-hydrodynamics SLM
(`Qwen/Qwen2.5-3B-Instruct` + LoRA in `grpo_run3_output/`).

**No Groq / `.env` API keys are required for the demo.**

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-app.txt
python -m app.gradio_app
```

Open http://127.0.0.1:7860

Optional env vars:

| Variable | Default | Meaning |
|----------|---------|---------|
| `ADAPTER_PATH` | `./grpo_run3_output` | LoRA adapter directory |
| `BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | HF base model |
| `MAX_NEW_TOKENS` | `512` | Generation length |
| `TEMPERATURE` | `0.25` | Sampling temperature (`0` = greedy) |
| `SERVER_PORT` | `7860` | Gradio port |
| `GRADIO_SHARE` | unset | Set `1` for a temporary public Gradio link |

Local CPU works but is slow for a 3B model. A CUDA GPU is strongly recommended.

### Deploy on Modal (free / cheap remote GPU)

Modal Starter includes monthly free compute credits and scales to **zero when idle**
(no charge while unused). A T4 is enough for this 3B + LoRA demo.

```bash
pip install modal
modal setup
modal deploy app/modal_app.py
```

For a temporary URL during development:

```bash
modal serve app/modal_app.py
```

The first cold start downloads the base model into a persistent Modal Volume
(`marine-hydro-hf-cache`); later starts reuse the cache.

### Fallback: Google Colab (free T4)

If Modal credits are exhausted:

1. Upload this repo (or at least `app/` + `grpo_run3_output/`) to Drive / clone it in Colab.
2. In a Colab runtime with a free T4 GPU:

```python
!pip install -r requirements-app.txt
import os
os.environ["GRADIO_SHARE"] = "1"
os.environ["ADAPTER_PATH"] = "/content/MTP/grpo_run3_output"  # adjust path
!python -m app.gradio_app
```

3. Open the printed `*.gradio.live` share link (temporary).

## Setup (research / training)

1. Create a Python 3.10+ environment and install dependencies used by your workflow (e.g. `transformers`, `trl`, `torch`, `peft`, `vllm` as needed for training or inference).
2. Copy `.env.example` to `.env` (or export variables in your shell) and set your API keys. **Do not commit `.env`.**

## Layout (high level)

- `app/` — Gradio chat demo + Modal deploy entrypoint.
- Root training/eval scripts and shared utilities (e.g. `physics_verifier.py`, `train_grpo.py`).
- `SFT2+GRPO4/`, `SFT3_NumAug/`, `GRPO5_LoRA_Expanded/` — experiment-specific pipelines and configs.
- `Report/`, `PPT/` — thesis report and slides (LaTeX).
- `documentation/` — run logs and design notes.

## Security

API keys must live only in environment variables or a local `.env` file that is git-ignored. If a key was ever committed or pasted into code, rotate it in the provider dashboard.
