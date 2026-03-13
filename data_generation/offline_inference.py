"""
大模型批量推理脚本 - 生成 SFT 训练数据

使用 Qwen3-VL-235B-A22B-Thinking-AWQ（或其他大模型）对数学题图像
进行批量推理，生成带 <think> 推理过程的回答，用于后续 SFT 训练。

用法：
    python offline_inference.py <input.jsonl> <output.jsonl> [image_dir]
"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import sys

# ==================== 配置 ====================
# 数据合成用大模型（需要 8×GPU，AWQ 量化）
# 下载地址: https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-AWQ
# 也可替换为其他大模型，如 Qwen2.5-VL-72B、GPT-4o 等
MODEL_PATH = "./Qwen3-VL-235B-A22B-Thinking-AWQ"
CONTEXT_LENGTH = 12000
TENSOR_PARALLEL_SIZE = 8
GPU_MEMORY_UTILIZATION = 0.8
MAX_NUM_SEQS = 128
SWAP_SPACE = 16
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 720
TEMPERATURE = 0.6
TOP_P = 0.001
REPETITION_PENALTY = 1.05
MAX_TOKENS = 12800
SYSTEM_PROMPT = "You are a math problem-solving assistant, please carefully analyze the problems in the image and provide detailed answers."
USER_PROMPT = "Please answer the questions in the image, and use \\boxed{} to enclose the final answer."


def load_jsonl(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                obj['modelprint'] = ''
                data.append(obj)
    return data


def preprocess_sample(obj, image_dir, processor):
    image_path = obj.get('image', '')
    if not os.path.isabs(image_path):
        image_path = os.path.join(image_dir, image_path)

    if not os.path.exists(image_path):
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path,
             "resized_width": IMAGE_WIDTH, "resized_height": IMAGE_HEIGHT},
            {"type": "text", "text": USER_PROMPT}
        ]}
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    mm_data = {"image": image_inputs} if image_inputs is not None else {}

    return {"prompt": prompt, "multi_modal_data": mm_data}


def main(image_dir, input_jsonl, output_jsonl, model_path=MODEL_PATH):
    print(f"模型: {model_path} | TP={TENSOR_PARALLEL_SIZE} | 图像: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")

    llm = LLM(
        model=model_path,
        max_model_len=CONTEXT_LENGTH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
        swap_space=SWAP_SPACE,
        trust_remote_code=True,
        disable_mm_preprocessor_cache=True,
        enable_expert_parallel=True,
        quantization="awq",
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY, max_tokens=MAX_TOKENS,
    )

    processor = AutoProcessor.from_pretrained(model_path)
    chat_template_path = os.path.join(model_path, "chat_template.json")
    if os.path.exists(chat_template_path):
        with open(chat_template_path, 'r', encoding='utf-8') as f:
            processor.chat_template = json.load(f)["chat_template"]

    input_data = load_jsonl(input_jsonl)
    print(f"加载 {len(input_data)} 个样本")

    valid_inputs, valid_objs = [], []
    for obj in tqdm(input_data, desc="预处理"):
        llm_input = preprocess_sample(obj, image_dir, processor)
        if llm_input:
            valid_inputs.append(llm_input)
            valid_objs.append(obj)

    outputs = llm.generate(valid_inputs, sampling_params=sampling_params)

    for obj, out in zip(valid_objs, outputs):
        obj["modelprint"] = out.outputs[0].text if out.outputs else ""

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in valid_objs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"完成: {len(valid_objs)}/{len(input_data)} 个样本 → {output_jsonl}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python offline_inference.py <input.jsonl> <output.jsonl> [image_dir]")
        sys.exit(1)
    main(sys.argv[3] if len(sys.argv) > 3 else "./", sys.argv[1], sys.argv[2])
