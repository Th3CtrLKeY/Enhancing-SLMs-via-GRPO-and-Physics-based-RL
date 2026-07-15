"""
Gradio chat UI for the Marine Hydrodynamics SLM.

Usage (local):
    pip install -r requirements-app.txt
    python -m app.gradio_app

Env:
    ADAPTER_PATH   — LoRA directory (default: ./grpo_run3_output)
    BASE_MODEL     — HF base model id (default: Qwen/Qwen2.5-3B-Instruct)
    MAX_NEW_TOKENS — generation length (default: 512)
    TEMPERATURE    — sampling temperature (default: 0.25)
    SERVER_PORT    — Gradio port (default: 7860)
"""

from __future__ import annotations

import os
from typing import List, Tuple

import gradio as gr

from app.inference import get_model

EXAMPLE_PROMPTS = [
    [
        "Explain the physical meaning of the Froude number and its significance "
        "in ship resistance prediction."
    ],
    [
        "A submarine travels at 5 m/s underwater. Its hull has a characteristic "
        "length of 80 m. The kinematic viscosity of seawater is 1.19e-6 m²/s. "
        "Calculate the Reynolds number and state whether the boundary layer is "
        "likely laminar or turbulent."
    ],
    [
        "Which of the following statements about the Froude number is CORRECT?\n\n"
        "Options:\n"
        "A: It is the ratio of inertial forces to gravitational forces and is dimensionless\n"
        "B: It is the ratio of viscous forces to inertial forces\n"
        "C: It has units of m/s\n"
        "D: It is only relevant for pipe flows, not free-surface flows"
    ],
    [
        "What is added mass in the context of marine hydrodynamics and why does it "
        "matter for vessel motion analysis?"
    ],
]


def _history_to_pairs(history: list) -> List[Tuple[str, str]]:
    """Normalize Gradio chatbot history into (user, assistant) pairs."""
    pairs: List[Tuple[str, str]] = []
    if not history:
        return pairs

    # Gradio 4+: list of {"role": ..., "content": ...}
    if isinstance(history[0], dict):
        user_buf = None
        for turn in history:
            role = turn.get("role")
            content = turn.get("content") or ""
            if role == "user":
                user_buf = content
            elif role == "assistant" and user_buf is not None:
                pairs.append((user_buf, content))
                user_buf = None
        return pairs

    # Gradio 3 style: list of [user, assistant]
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((item[0] or "", item[1] or ""))
    return pairs


def respond(message: str, history: list) -> str:
    message = (message or "").strip()
    if not message:
        return "Please enter a marine hydrodynamics question."

    model = get_model()
    prior = _history_to_pairs(history)
    return model.generate(message, history=prior)


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Marine Hydrodynamics SLM",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 900px !important; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # Marine Hydrodynamics SLM
            Ask conceptual, numerical, or multiple-choice questions about marine
            hydrodynamics and ocean engineering.

            Powered by **Qwen2.5-3B-Instruct** + a GRPO LoRA adapter from this thesis repo.
            No API keys are required for this demo.
            """
        )
        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
            show_copy_button=True,
            type="messages",
        )
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask a hydrodynamics question…",
                show_label=False,
                scale=4,
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear = gr.ClearButton([msg, chatbot], value="Clear")

        gr.Examples(
            examples=EXAMPLE_PROMPTS,
            inputs=msg,
            label="Example prompts",
        )

        def user_submit(user_message, chat_history):
            user_message = (user_message or "").strip()
            if not user_message:
                return "", chat_history
            chat_history = list(chat_history or [])
            chat_history.append({"role": "user", "content": user_message})
            return "", chat_history

        def bot_reply(chat_history):
            chat_history = list(chat_history or [])
            if not chat_history:
                return chat_history
            # Build history excluding the latest user turn for context.
            prior = chat_history[:-1]
            question = chat_history[-1]["content"]
            answer = respond(question, prior)
            chat_history.append({"role": "assistant", "content": answer})
            return chat_history

        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_reply, chatbot, chatbot
        )
        send.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_reply, chatbot, chatbot
        )

        # Warm the model on startup when possible (local / Modal).
        demo.load(lambda: get_model() and None, None, None)

    return demo


def main() -> None:
    port = int(os.environ.get("SERVER_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"}
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=os.environ.get("SERVER_NAME", "0.0.0.0"),
        server_port=port,
        share=share,
    )


if __name__ == "__main__":
    main()
