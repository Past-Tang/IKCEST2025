"""
三阶段推理流水线主入口

流程：
    阶段1（OCR）：InternVL3.5-2B 识别题目文本 + 图表检测
    阶段2（文本推理）：OpenMath-Nemotron-1.5B 处理纯文字题
    阶段3（视觉推理）：InternVL3.5-2B 处理含图表题 + 合并结果

使用方法：
    完整流程: python run.py <image_dir> <input_jsonl> <output_jsonl>
    阶段1:    python run.py stage1 <image_dir> <input_jsonl> <ocr_output.jsonl>
    阶段2:    python run.py stage2 <image_dir> <ocr.jsonl> <text_output.jsonl>
    阶段3:    python run.py stage3 <image_dir> <ocr.jsonl> <text.jsonl> <output.jsonl>
"""

import os
import logging
vllm_utils_logger = logging.getLogger("vllm.utils")
vllm_utils_logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=".*intended overrides are not keyword args.*")
import json
import sys
from prompt import get_text_prompt, get_vl_prompt
from answer import MathAnswerExtractor
from ocr import OCRModel
import text
import vl
import torch
import gc

answer_extractor = MathAnswerExtractor()

# ==================== 全局配置 ====================
# 模型下载地址（HuggingFace）：
#   文本模型: https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B
#   视觉模型: https://huggingface.co/OpenGVLab/InternVL3-2B-Instruct
TEXT_MODEL_PATH = "./OpenMath-Nemotron-1.5B"          # nvidia/OpenMath-Nemotron-1.5B
VL_MODEL_PATH = "./InternVL3_5-2B-Instruct"           # OpenGVLab/InternVL3-2B-Instruct
OCR_MODEL_PATH = "./InternVL3_5-2B-Instruct"          # 与 VL 共用同一模型
MAX_TOKENS_TEXT = 13384
MAX_TOKENS_VL = 14384
# ================================================


def load_jsonl(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_default_answer(question_type: str) -> str:
    if question_type == "选择题":
        return "C"
    else:
        return "1.000000"


def _normalize_to_binary_type(question_type: str) -> str:
    """将题型归一为二元类型：'选择题' 或 '填空题'（非选择题）。"""
    if not question_type:
        return "填空题"
    q = str(question_type).strip()
    lower_q = q.lower()
    choice_markers = [
        "选择", "单选", "多选", "choice", "multiple choice", "mcq"
    ]
    if any(marker in q for marker in choice_markers) or any(marker in lower_q for marker in choice_markers):
        return "选择题"
    return "填空题"


def resolve_question_type(item: dict) -> tuple:
    """
    解析题型，优先使用 tag 字段，回退到 type/question_type/qtype/category。
    返回 (prompt_type, extraction_type)
    """
    raw = None
    for key in ("tag", "type", "question_type", "qtype", "category"):
        if key in item and item[key]:
            raw = str(item[key]).strip()
            break
    if not raw:
        raw = "填空题"

    if raw in ("选择题", "填空题", "计算应用题"):
        prompt_type = raw
    else:
        prompt_type = "选择题" if _normalize_to_binary_type(raw) == "选择题" else "填空题"

    extraction_type = _normalize_to_binary_type(prompt_type)
    return prompt_type, extraction_type


def inference_text_model(items, model_path):
    """使用文本模型对无图表题目进行批量推理"""
    if not items:
        return items

    prompts_list = []
    for item in items:
        ocr_text = item['ocr_result']['text']
        prompt_type, _ = resolve_question_type(item)
        item['question_type'] = prompt_type

        prompt_template = get_text_prompt(prompt_type)
        messages = [
            {"role": "system", "content": prompt_template['system']},
            {"role": "user", "content": f"{prompt_template['user']}\n\n{ocr_text}"}
        ]
        prompts_list.append(messages)

    print(f"文本模型批量推理中...")
    answers = text.generate_answers(model_path, prompts_list, max_tokens=MAX_TOKENS_TEXT)

    for i, item in enumerate(items):
        item['model_output'] = answers[i]
        _, extraction_type = resolve_question_type(item)
        extracted_answer, status = answer_extractor.extract_answer(answers[i], extraction_type)

        if extracted_answer is None or status != "成功":
            extracted_answer = get_default_answer(extraction_type)
            status = "使用默认答案"

        item['answer'] = extracted_answer
        item['extraction_status'] = status

    return items


def inference_vision_model(items, image_dir, model_path):
    """使用视觉模型对有图表题目进行批量推理"""
    if not items:
        return items

    inputs_list = []
    for item in items:
        image_path = os.path.join(image_dir, item['image'])
        prompt_type, _ = resolve_question_type(item)
        item['question_type'] = prompt_type

        prompt_template = get_vl_prompt(prompt_type)
        question = f"{prompt_template['system']}\n\n{prompt_template['user']}".strip()
        inputs_list.append((image_path, question))

    print(f"视觉模型批量推理中...")
    answers = vl.generate_answers(model_path, inputs_list, max_tokens=MAX_TOKENS_VL)

    for i, item in enumerate(items):
        item['model_output'] = answers[i]
        _, extraction_type = resolve_question_type(item)
        extracted_answer, status = answer_extractor.extract_answer(answers[i], extraction_type)

        if extracted_answer is None or status != "成功":
            extracted_answer = get_default_answer(extraction_type)
            status = "使用默认答案"

        item['answer'] = extracted_answer
        item['extraction_status'] = status

    return items


def _release_gpu_memory():
    """彻底释放 GPU 显存"""
    import time
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(3)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage1_ocr(image_dir, input_jsonl, ocr_output_jsonl):
    """阶段1: OCR识别并保存结果"""
    print("=" * 80)
    print("【阶段1: OCR识别（vLLM 批量推理）】")
    print("=" * 80)

    input_data = load_jsonl(input_jsonl)
    print(f"加载了 {len(input_data)} 条数据")

    ocr_model = OCRModel(model_id=OCR_MODEL_PATH)
    all_image_paths = [os.path.join(image_dir, item['image']) for item in input_data]

    print(f"开始批量OCR推理（共 {len(all_image_paths)} 张图像）...")
    all_ocr_results = ocr_model.ocr_batch(all_image_paths)

    results = []
    for item, ocr_result in zip(input_data, all_ocr_results):
        item['ocr_result'] = ocr_result
        results.append(item)

    print("\n释放OCR模型显存...")
    try:
        ocr_model.release()
        del ocr_model
        _release_gpu_memory()
        print("OCR模型显存释放完成")
    except Exception as e:
        print(f"释放OCR模型资源时出错: {e}")

    save_jsonl(results, ocr_output_jsonl)
    print(f"\nOCR结果已保存到: {ocr_output_jsonl}")

    no_chart = sum(1 for item in results if not item.get('ocr_result', {}).get('is_chart', False))
    chart = sum(1 for item in results if item.get('ocr_result', {}).get('is_chart', False))
    print(f"无图表题目: {no_chart} 条 | 有图表题目: {chart} 条")


def stage2_text_inference(image_dir, ocr_jsonl, text_output_jsonl):
    """阶段2: 文本模型推理（无图表题目）"""
    print("=" * 80)
    print("【阶段2: 文本模型推理】")
    print("=" * 80)

    all_data = load_jsonl(ocr_jsonl)
    no_chart_items = [item for item in all_data
                     if not item.get('ocr_result', {}).get('is_chart', False)]
    print(f"无图表题目: {len(no_chart_items)} 条")

    if no_chart_items:
        no_chart_items = inference_text_model(no_chart_items, TEXT_MODEL_PATH)
        print("\n释放文本模型显存...")
        _release_gpu_memory()
        save_jsonl(no_chart_items, text_output_jsonl)
        print(f"文本模型结果已保存，共 {len(no_chart_items)} 条")
    else:
        print("没有无图表题目，跳过")
        save_jsonl([], text_output_jsonl)


def stage3_vision_inference(image_dir, ocr_jsonl, text_jsonl, final_output_jsonl):
    """阶段3: 视觉模型推理（有图表题目）+ 合并最终结果"""
    print("=" * 80)
    print("【阶段3: 视觉模型推理与结果合并】")
    print("=" * 80)

    all_data = load_jsonl(ocr_jsonl)
    text_results = load_jsonl(text_jsonl)
    text_dict = {item.get('id', item.get('image')): item for item in text_results}

    chart_items = [item for item in all_data
                  if item.get('ocr_result', {}).get('is_chart', False)]
    print(f"有图表题目: {len(chart_items)} 条")

    if chart_items:
        chart_items = inference_vision_model(chart_items, image_dir, VL_MODEL_PATH)
        print("\n释放视觉模型显存...")
        _release_gpu_memory()

    final_results = []
    for item in all_data:
        item_id = item.get('id', item.get('image'))
        if item_id in text_dict:
            final_results.append(text_dict[item_id])
        else:
            chart_item = next((ci for ci in chart_items
                             if ci.get('id', ci.get('image')) == item_id), None)
            final_results.append(chart_item if chart_item else item)

    if final_results and 'id' in final_results[0]:
        final_results.sort(key=lambda x: x.get('id', 0))

    save_jsonl(final_results, final_output_jsonl)
    print(f"\n最终结果已保存到: {final_output_jsonl}，共 {len(final_results)} 条")


def main(image_dir, input_jsonl, output_jsonl):
    """完整流程运行"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f1:
        ocr_path = f1.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f2:
        text_path = f2.name
    try:
        stage1_ocr(image_dir, input_jsonl, ocr_path)
        stage2_text_inference(image_dir, ocr_path, text_path)
        stage3_vision_inference(image_dir, ocr_path, text_path, output_jsonl)
    finally:
        for p in [ocr_path, text_path]:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  完整流程: python run.py <image_dir> <input_jsonl> <output_jsonl>")
        print("  阶段1:    python run.py stage1 <image_dir> <input_jsonl> <ocr_output.jsonl>")
        print("  阶段2:    python run.py stage2 <image_dir> <ocr.jsonl> <text_output.jsonl>")
        print("  阶段3:    python run.py stage3 <image_dir> <ocr.jsonl> <text.jsonl> <output.jsonl>")
        sys.exit(1)

    if sys.argv[1] == "stage1":
        stage1_ocr(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "stage2":
        stage2_text_inference(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "stage3":
        stage3_vision_inference(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
