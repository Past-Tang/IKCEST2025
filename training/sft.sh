#!/bin/bash
# ===========================================================
# SFT 训练脚本 - Qwen2.5-VL-3B-Instruct
#
# 配置：8 GPU / DeepSpeed ZeRO-3 / 冻结 ViT / bfloat16
# 系统提示词强制 <think></think> + \boxed{} 输出格式
# ===========================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 基座模型下载: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model ./Qwen2.5-VL-3B-Instruct \
    --model_type qwen2_5_vl \
    --system "You first analyze both the image and the question, then carefully reason through the problem step by step. You must verify the correctness of each reasoning step. If errors or inconsistencies are found, you reflect and correct them. The reasoning process is enclosed in <think> </think> tags, followed by optional description, then the final answer enclosed in \\boxed{}. Format: <think>reasoning process with reflections and corrections if needed</think>Optional description. \\boxed{final answer}" \
    --max_pixels '570752' \
    --truncation_strategy delete \
    --train_type full \
    --dataset ./sft_dataset.jsonl \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 1.5 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 16 \
    --eval_steps 25 \
    --save_steps 25 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 11520 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed ./ds_config_zero3.json \
    --gradient_checkpointing true
