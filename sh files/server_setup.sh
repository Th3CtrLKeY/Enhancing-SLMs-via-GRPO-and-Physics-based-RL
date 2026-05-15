#!/usr/bin/env bash
# =============================================================================
#  server_setup.sh — MTP QA Generation Server Setup
#  Run this script on the H100 server (10.71.9.8) once.
#
#  What it does:
#    1. Creates conda env "mtp_qa" with Python 3.11
#    2. Installs vLLM + dependencies
#    3. Downloads Qwen2.5-72B-Instruct-AWQ (4-bit, ~42 GB)
#    4. Launches vLLM as a background server on port 8000
# =============================================================================

set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-72B-Instruct-AWQ"
ENV_NAME="mtp_qa"
PORT=8000
LOG_FILE="$HOME/vllm_server.log"
MTP_DIR="$HOME/mtp"

echo "======================================================"
echo "  MTP QA Generation — Server Setup"
echo "  Model : $MODEL_ID"
echo "  Port  : $PORT"
echo "======================================================"

# ── 0. Create project directory ──────────────────────────────────────────────
mkdir -p "$MTP_DIR"
echo "[0] Project dir: $MTP_DIR"

# ── 1. Create conda environment ───────────────────────────────────────────────
echo ""
echo "[1] Creating conda environment '$ENV_NAME' with Python 3.11..."

if conda env list | grep -q "^$ENV_NAME "; then
    echo "    [SKIP] Environment '$ENV_NAME' already exists."
else
    conda create -y -n "$ENV_NAME" python=3.11
    echo "    [OK] Environment created."
fi

# Activate env for subsequent installs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 2. Install vLLM ───────────────────────────────────────────────────────────
echo ""
echo "[2] Installing vLLM (this may take a few minutes)..."

pip install --upgrade pip --quiet
pip install vllm --quiet
pip install huggingface_hub tqdm requests --quiet

echo "    [OK] vLLM installed: $(python -c 'import vllm; print(vllm.__version__)')"

# ── 3. Download the model ─────────────────────────────────────────────────────
echo ""
echo "[3] Downloading model: $MODEL_ID"
echo "    (This will take 20–40 min on first run; subsequent runs skip this)"

python - <<'EOF'
from huggingface_hub import snapshot_download
import os

model_id = "Qwen/Qwen2.5-72B-Instruct-AWQ"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print(f"  Downloading to: {cache_dir}")
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    ignore_patterns=["*.pt", "*.bin"],   # prefer safetensors
)
print("  [OK] Model downloaded.")
EOF

# ── 4. Kill any existing vLLM server on port 8000 ────────────────────────────
echo ""
echo "[4] Checking for existing server on port $PORT..."
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo "    Killing existing process on port $PORT..."
    kill -9 $(lsof -ti:$PORT) || true
    sleep 2
fi

# ── 5. Launch vLLM server ─────────────────────────────────────────────────────
echo ""
echo "[5] Launching vLLM server in background..."
echo "    Logs → $LOG_FILE"

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --port "$PORT" \
    --host "0.0.0.0" \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "    [OK] vLLM launched (PID=$SERVER_PID)"

# ── 6. Wait for server to be ready ───────────────────────────────────────────
echo ""
echo "[6] Waiting for server to come online (up to 3 minutes)..."
MAX_WAIT=180
ELAPSED=0

while ! curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo -ne "    waited ${ELAPSED}s / ${MAX_WAIT}s...\r"
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo ""
        echo "    [ERROR] Server did not come up in time. Check logs:"
        echo "            tail -50 $LOG_FILE"
        exit 1
    fi
done

echo ""
echo "    [OK] Server is up and healthy!"

# ── 7. Quick smoke test ───────────────────────────────────────────────────────
echo ""
echo "[7] Smoke test — sending one prompt..."

RESPONSE=$(curl -sf "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages": [{"role":"user","content":"What is the Reynolds number?"}],
    "max_tokens": 100
  }')

if echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); print('  Response:', d['choices'][0]['message']['content'][:120])"; then
    echo "    [OK] Smoke test passed!"
else
    echo "    [WARN] Unexpected response. Check logs: tail -f $LOG_FILE"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Setup Complete!"
echo "  vLLM endpoint : http://localhost:$PORT/v1"
echo "  Model PID     : $SERVER_PID"
echo "  Logs          : $LOG_FILE"
echo ""
echo "  Next step: python ~/mtp/generate_qa.py"
echo "======================================================"
