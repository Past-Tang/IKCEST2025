#!/bin/bash
# ===========================================================
# 启动 vLLM rollout 服务（GRPO 训练时需要）
#
# 使用 2 GPU 运行推理服务，其余 GPU 用于训练
# ===========================================================

export CUDA_VISIBLE_DEVICES=0,1
export MAX_PIXELS=570752

# 基座模型下载: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
swift rollout \
    --model ./Qwen2.5-VL-3B-Instruct \
    --model_type qwen2_5_vl \
    --vllm_data_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.95 \
    --vllm_enable_prefix_caching \
    --vllm_limit_mm_per_prompt '{"image": 1, "video": 1}' \
    --vllm_max_model_len 16000 \
    --port 8000 \
    --temperature 0.6
