#!/usr/bin/env bash
# launch_vllm.sh — run this on the server to start vLLM
# Free VRAM: ~64 GB (2 other users using 31 GB on the H100)
# Qwen2.5-72B-AWQ needs ~42 GB → use 0.65 utilization (~62 GB)

pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3

nohup ~/mtp/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.65 \
    --max-num-seqs 16 \
    --port 8000 \
    --host 0.0.0.0 \
    > ~/mtp/vllm_server.log 2>&1 &

echo "vLLM started with PID=$!"
echo "Watch logs with:  tail -f ~/mtp/vllm_server.log"
echo "Health check:     curl http://localhost:8000/health"
