#!/bin/bash
# ===========================================================
# 三阶段推理运行脚本（竞赛环境）
#
# V100-32GB 兼容性配置：
#   - 禁用 Triton Flash Attention（V100 不支持）
#   - 使用 XFormers 替代
#   - 禁用 torch.compile（V100 性能下降）
# ===========================================================

# V100 vLLM 兼容性环境变量
export USE_MEMORY_EFFICIENT_ATTENTION=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_CONFIGURE_LOGGING=0
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_USE_TRITON_FLASH_ATTN=0
export TRITON_INTERPRET=0
export VLLM_USE_TRITON_AWQ=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==================== 路径配置 ====================
DEFAULT_IMAGE_DIR="./data/images"
DEFAULT_QUERY_FILE="./data/input.jsonl"
DEFAULT_OUTPUT_FILE="./output.jsonl"

IMAGE_INPUT_DIR=${IMAGE_INPUT_DIR:-$DEFAULT_IMAGE_DIR}
QUERY_PATH=${QUERY_PATH:-$DEFAULT_QUERY_FILE}
OUPUT_PATH=${OUPUT_PATH:-$DEFAULT_OUTPUT_FILE}

OCR_RESULT=./ocr_result.jsonl
TEXT_RESULT=./text_result.jsonl

# ==================== 阶段1: OCR识别 ====================
echo "=========================================="
echo "【阶段1/3: OCR识别】"
echo "=========================================="
python run.py stage1 $IMAGE_INPUT_DIR $QUERY_PATH $OCR_RESULT
if [ $? -ne 0 ]; then echo "阶段1失败"; exit 1; fi

# 阶段2 使用文本模型，可解除部分限制以加速
unset VLLM_TORCH_COMPILE_LEVEL
unset TRITON_INTERPRET

# ==================== 阶段2: 文本模型推理 ====================
echo ""
echo "=========================================="
echo "【阶段2/3: 文本模型推理】"
echo "=========================================="
python run.py stage2 $IMAGE_INPUT_DIR $OCR_RESULT $TEXT_RESULT
if [ $? -ne 0 ]; then echo "阶段2失败"; exit 1; fi

# 阶段3 恢复 V100 兼容性配置
export VLLM_TORCH_COMPILE_LEVEL=0
export TRITON_INTERPRET=0

# ==================== 阶段3: 视觉模型推理 ====================
echo ""
echo "=========================================="
echo "【阶段3/3: 视觉模型推理】"
echo "=========================================="
python run.py stage3 $IMAGE_INPUT_DIR $OCR_RESULT $TEXT_RESULT $OUPUT_PATH
if [ $? -ne 0 ]; then echo "阶段3失败"; exit 1; fi

echo ""
echo "=========================================="
echo "全部阶段完成！最终结果: $OUPUT_PATH"
echo "=========================================="
