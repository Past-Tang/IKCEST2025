"""
Baseline 方案 - Qwen2.5-VL-3B 单模型端到端推理

所有题目直接用一个 VL 模型处理（不做 OCR 分流），得分 55.7~65.1。
作为对比参考，展示三阶段流水线相对于单模型方案的提升。

用法：
    python run.py <image_dir> <input.jsonl> <output.jsonl>
"""

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import sys
from prompt import get_prompt
from answer import MathAnswerExtractor

answer_extractor = MathAnswerExtractor()

# 下载地址: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
MODEL_PATH = "./Qwen2.5-VL-3B-Instruct"
MAX_TOKENS = 16384
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 720


def get_default_answer(question_type):
    return "C" if question_type == "选择题" else "1.000000"


def process_model_response(response, question_type):
    try:
        answer, status = answer_extractor.extract_answer(response, question_type)
        if status != "成功" or answer is None:
            answer = get_default_answer(question_type)
        return {"step": "", "formatted_answer": answer}
    except Exception:
        return {"step": "", "formatted_answer": get_default_answer(question_type)}


def load_jsonl(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def preprocess_sample(obj, image_dir, processor):
    image_path = os.path.join(image_dir, obj['image'])
    if not os.path.exists(image_path):
        return None

    question_type = obj.get("tag", "")
    prompt = get_prompt(question_type)

    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path,
             "resized_width": IMAGE_WIDTH, "resized_height": IMAGE_HEIGHT},
            {"type": "text", "text": prompt["user"]}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    mm_data = {"image": image_inputs} if image_inputs else {}

    return {"prompt": text, "multi_modal_data": mm_data}


def main(image_dir, input_jsonl, output_jsonl):
    llm = LLM(
        model=MODEL_PATH, max_model_len=32768,
        gpu_memory_utilization=0.90, disable_mm_preprocessor_cache=True,
        tensor_parallel_size=1, trust_remote_code=True, max_num_seqs=8
    )

    sampling_params = SamplingParams(
        temperature=0.2, top_p=0.001,
        repetition_penalty=1.12, max_tokens=MAX_TOKENS,
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    chat_template_path = os.path.join(MODEL_PATH, "chat_template.json")
    if os.path.exists(chat_template_path):
        with open(chat_template_path, 'r') as f:
            processor.chat_template = json.load(f)["chat_template"]

    input_data = load_jsonl(input_jsonl)
    valid_inputs, valid_objs = [], []

    for obj in tqdm(input_data, desc="预处理"):
        result = preprocess_sample(obj, image_dir, processor)
        if result:
            valid_inputs.append(result)
            valid_objs.append(obj)

    outputs = llm.generate(valid_inputs, sampling_params=sampling_params)

    results = []
    for obj, out in zip(valid_objs, outputs):
        text = out.outputs[0].text if out.outputs else ""
        result = process_model_response(text, obj.get("tag", ""))
        obj["answer"] = result["formatted_answer"]
        results.append(obj)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"结果已保存到: {output_jsonl}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
