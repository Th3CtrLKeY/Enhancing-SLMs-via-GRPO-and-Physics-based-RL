# Enhancing SLMs via GRPO and Physics-based RL

Research codebase for supervised fine-tuning (SFT), Group Relative Policy Optimization (GRPO), and physics-grounded reward shaping on small language models.

## Setup

1. Create a Python 3.10+ environment and install dependencies used by your workflow (e.g. `transformers`, `trl`, `torch`, `peft`, `vllm` as needed for training or inference).
2. Copy `.env.example` to `.env` (or export variables in your shell) and set your API keys. **Do not commit `.env`.**

## Layout (high level)

- Root training/eval scripts and shared utilities (e.g. `physics_verifier.py`, `train_grpo.py`).
- `SFT2+GRPO4/`, `SFT3_NumAug/`, `GRPO5_LoRA_Expanded/` — experiment-specific pipelines and configs.
- `Report/`, `PPT/` — thesis report and slides (LaTeX).
- `documentation/` — run logs and design notes.

## Security

API keys must live only in environment variables or a local `.env` file that is git-ignored. If a key was ever committed or pasted into code, rotate it in the provider dashboard.
