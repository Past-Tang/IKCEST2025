
#!/bin/bash


# 配置 vLLM 以避免 V100 GPU 的 Triton 兼容性问题
export USE_MEMORY_EFFICIENT_ATTENTION=1                     # 使用内存高效的注意力机制
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1                     # 禁用警告信息
export VLLM_CONFIGURE_LOGGING=0                             # 禁用日志配置          
export VLLM_USE_V1=0                                    # 禁用 V1 引擎，使用 V0 引擎
export VLLM_ATTENTION_BACKEND=XFORMERS                  # 使用 XFormers 后端
export VLLM_TORCH_COMPILE_LEVEL=0                       # 禁用 torch.compile
export VLLM_USE_TRITON_FLASH_ATTN=0                     # 禁用 Triton Flash Attention
export TRITON_INTERPRET=0                               # 禁用 Triton 解释器
export VLLM_USE_TRITON_AWQ=0                            # 禁用 Triton AWQ
export VLLM_WORKER_MULTIPROC_METHOD=spawn               # 使用 spawn 方法
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 优化显存分配

# ==================== 路径配置 ====================
# 【本地测试】默认路径配置（按实际情况修改）
DEFAULT_IMAGE_DIR="./data/images"
DEFAULT_QUERY_FILE="./data/input.jsonl"
DEFAULT_OUTPUT_FILE="./output.jsonl"


# 【竞赛环境】评测系统会自动设置以下变量，本地测试时使用默认值
IMAGE_INPUT_DIR=${IMAGE_INPUT_DIR:-$DEFAULT_IMAGE_DIR}
QUERY_PATH=${QUERY_PATH:-$DEFAULT_QUERY_FILE}
OUPUT_PATH=${OUPUT_PATH:-$DEFAULT_OUTPUT_FILE}

# 中间结果文件路径
OCR_RESULT=./ocr_result.jsonl
TEXT_RESULT=./text_result.jsonl

# ==================== 三阶段运行 ====================
# 阶段1: OCR识别
echo "=========================================="
echo "【阶段1/3: OCR识别】"
echo "=========================================="
python run.py stage1 $IMAGE_INPUT_DIR $QUERY_PATH $OCR_RESULT
if [ $? -ne 0 ]; then
    echo "❌ 阶段1失败"
    exit 1
fi
# 配置 加速
unset VLLM_TORCH_COMPILE_LEVEL
unset TRITON_INTERPRET

# 阶段2: 文本模型推理（无图表题）
echo ""
echo "=========================================="
echo "【阶段2/3: 文本模型推理】"
echo "=========================================="
python run.py stage2 $IMAGE_INPUT_DIR $OCR_RESULT $TEXT_RESULT
if [ $? -ne 0 ]; then
    echo "❌ 阶段2失败"
    exit 1
fi
# 配置 vLLM 以避免 V100 GPU 的 Triton 兼容性问题
export USE_MEMORY_EFFICIENT_ATTENTION=1                     # 使用内存高效的注意力机制
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1                     # 禁用警告信息
export VLLM_CONFIGURE_LOGGING=0                             # 禁用日志配置          
export VLLM_USE_V1=0                                    # 禁用 V1 引擎，使用 V0 引擎
export VLLM_ATTENTION_BACKEND=XFORMERS                  # 使用 XFormers 后端
export VLLM_TORCH_COMPILE_LEVEL=0                       # 禁用 torch.compile
export VLLM_USE_TRITON_FLASH_ATTN=0                     # 禁用 Triton Flash Attention
export TRITON_INTERPRET=0                               # 禁用 Triton 解释器
export VLLM_USE_TRITON_AWQ=0                            # 禁用 Triton AWQ
export VLLM_WORKER_MULTIPROC_METHOD=spawn               # 使用 spawn 方法
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 优化显存分配
# 阶段3: 视觉模型推理（有图表题）+ 合并结果
echo ""
echo "=========================================="
echo "【阶段3/3: 视觉模型推理】"
echo "=========================================="
python run.py stage3 $IMAGE_INPUT_DIR $OCR_RESULT $TEXT_RESULT $OUPUT_PATH
if [ $? -ne 0 ]; then
    echo "❌ 阶段3失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 全部阶段完成！"
echo "最终结果: $OUPUT_PATH"
echo "=========================================="