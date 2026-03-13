"""
LLM 答案验证脚本

用大模型判断 offline_inference 生成的 \\boxed{} 答案
是否与参考答案一致（CONSISTENT / INCONSISTENT / UNCERTAIN）。

用法：
    python verify_answers.py <input.jsonl> <output.jsonl>
"""

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import os
import json
import sys
import re

# ==================== 配置 ====================
# 答案验证模型（推荐使用大参数文本模型以保证验证准确性）
# 可选: Qwen/Qwen3-235B-A22B-Instruct-AWQ, deepseek-ai/DeepSeek-R1 等
MODEL_PATH = "./verification_model"
CONTEXT_LENGTH = 12000
TENSOR_PARALLEL_SIZE = 8
GPU_MEMORY_UTILIZATION = 0.8
MAX_NUM_SEQS = 128
SWAP_SPACE = 16

VERIFY_SYSTEM_PROMPT = """You are a math answer verification assistant. Compare two mathematical answers and determine if they are equivalent.

Consider:
- Numerical equivalence (e.g., 15.0 = 15 = 15.000000)
- Mathematical equivalence (e.g., 1/2 = 0.5, √4 = 2)
- Ignore minor formatting differences

Respond with ONLY one word: Yes or No"""

VERIFY_USER_TEMPLATE = """Reference Answer: {reference_answer}
Model Answer: {model_answer}

Are these answers equivalent? (Yes/No)"""


def extract_last_boxed(text):
    if not text:
        return None
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def main(input_jsonl, output_jsonl, model_path=MODEL_PATH):
    llm = LLM(
        model=model_path, max_model_len=CONTEXT_LENGTH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=MAX_NUM_SEQS, swap_space=SWAP_SPACE,
        trust_remote_code=True, enable_expert_parallel=True,
    )

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=4096)
    processor = AutoProcessor.from_pretrained(model_path)

    chat_template_path = os.path.join(model_path, "chat_template.json")
    if os.path.exists(chat_template_path):
        with open(chat_template_path, 'r', encoding='utf-8') as f:
            processor.chat_template = json.load(f)["chat_template"]

    data = load_jsonl(input_jsonl)
    valid_inputs, valid_objs, skipped = [], [], []

    for obj in data:
        ref = obj.get('answer', '').strip()
        model_answer = extract_last_boxed(obj.get('modelprint', ''))

        if not ref or model_answer is None:
            obj['verification_result'] = 'SKIPPED'
            skipped.append(obj)
            continue

        prompt = processor.apply_chat_template([
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": VERIFY_USER_TEMPLATE.format(
                reference_answer=ref, model_answer=model_answer)}
        ], tokenize=False, add_generation_prompt=True, thinking_budget=512)

        valid_inputs.append({"prompt": prompt})
        obj['extracted_model_answer'] = model_answer
        valid_objs.append(obj)

    if valid_inputs:
        outputs = llm.generate(valid_inputs, sampling_params=sampling_params)
        for obj, out in zip(valid_objs, outputs):
            text = out.outputs[0].text.strip().upper()
            if "YES" in text:
                obj['verification_result'] = "CONSISTENT"
            elif "NO" in text:
                obj['verification_result'] = "INCONSISTENT"
            else:
                obj['verification_result'] = "UNCERTAIN"

    all_results = valid_objs + skipped
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    consistent = sum(1 for r in all_results if r.get('verification_result') == 'CONSISTENT')
    print(f"验证结果: CONSISTENT={consistent} / 总计={len(all_results)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python verify_answers.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
