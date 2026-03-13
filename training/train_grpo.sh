#!/bin/bash
# ===========================================================
# GRPO 训练脚本 - 基于 SFT 检查点继续训练
#
# 需要先启动 vLLM rollout 服务（见 start_vllm.sh）
# 使用 orm.py 中的 custom_math_reward 作为奖励函数
#
# 配置：6 GPU / DeepSpeed ZeRO-3 / 4 rollout / vLLM server
# ===========================================================

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export NPROC_PER_NODE=6
export MASTER_PORT=29600
export MAX_PIXELS=570752
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 基座模型下载: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
swift rlhf \
    --rlhf_type grpo \
    --model ./Qwen2.5-VL-3B-Instruct \
    --model_type qwen2_5_vl \
    --external_plugins ./orm.py \
    --reward_funcs custom_math_reward \
    --reward_weights 1.0 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner false \
    --dataset ./rl_dataset.jsonl \
    --split_dataset_ratio 0.03 \
    --load_from_cache_file true \
    --max_completion_length 16000 \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --max_grad_norm 0.5 \
    --beta 0.01 \
    --num_generations 4 \
    --temperature 0.7 \
    --repetition_penalty 1.12 \
    --num_iterations 2 \
    --async_generate true \
    --save_steps 25 \
    --eval_steps 50 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir ./output/GRPO \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed ./ds_config_zero3.json \
    --gradient_checkpointing true \
    --system "The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning are enclosed within <think> </think> tags, i.e., <think> reasoning process here, with potential reflections and corrections </think> final answer here, with the key result enclosed in \\boxed{} ." \
    --log_completions true

# ===========================================================
# 使用说明：
# 1. 先启动 vLLM 服务: bash start_vllm.sh
# 2. 再运行训练: bash train_grpo.sh
# ===========================================================
