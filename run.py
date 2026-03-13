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
from ocr import OCRModel  # 导入OCR模型
import text  # 文本模型推理
import vl    # 视觉模型推理
import torch
import gc

answer_extractor = MathAnswerExtractor()

# ==================== 全局配置 ====================
# 模型配置
TEXT_MODEL_PATH = "./OpenMath-Nemotron-1.5B"  # 文本模型路径
VL_MODEL_PATH = "./InternVL3_5-2B-Instruct"  # 视觉模型路径（使用 Ovis2-2B）
OCR_MODEL_PATH = "./InternVL3_5-2B-Instruct" # OCR模型路径（使用 Ovis2-2B）
MAX_TOKENS_TEXT = 13384  # 文本模型最大token
MAX_TOKENS_VL = 14384     # 视觉模型最大token
# ================================================

# 加载JSONL文件
def load_jsonl(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 保存JSONL文件
def save_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 获取题型默认答案
def get_default_answer(question_type: str) -> str:
    """
    获取题型的默认答案（当提取失败时使用）
    
    Args:
        question_type: 题型，如 "选择题"、"填空题"、"计算应用题"
    
    Returns:
        str: 默认答案
    """
    if question_type == "选择题":
        return "C"
    else:
        return "1.000000"

# 题型解析/标准化
def _normalize_to_binary_type(question_type: str) -> str:
    """将题型归一为二元类型：'选择题' 或 '填空题'（非选择题）。"""
    if not question_type:
        return "填空题"
    q = str(question_type).strip()
    lower_q = q.lower()
    # 选择题关键词（中英文与常见缩写）
    choice_markers = [
        "选择", "单选", "多选", "choice", "multiple choice", "mcq"
    ]
    if any(marker in q for marker in choice_markers) or any(marker in lower_q for marker in choice_markers):
        return "选择题"
    return "填空题"

def resolve_question_type(item: dict) -> tuple:
    """
    解析题型：
    - 优先使用 `tag`
    - 回退到 `type`、`question_type`、`qtype`、`category`
    返回 (prompt_type, extraction_type)
      - prompt_type: 用于 prompt 的原始类型（'选择题' / '填空题' / '计算应用题'）
      - extraction_type: 二元类型（'选择题' 或 '填空题'）用于答案提取与默认答案
    """
    raw = None
    for key in ("tag", "type", "question_type", "qtype", "category"):
        if key in item and item[key]:
            raw = str(item[key]).strip()
            break
    if not raw:
        raw = "填空题"

    # prompt_type 保留三类之一，其他统一为 '填空题'
    if raw in ("选择题", "填空题", "计算应用题"):
        prompt_type = raw
    else:
        # 简单兜底：如果包含“选/choice”等关键词，当作选择题
        prompt_type = "选择题" if _normalize_to_binary_type(raw) == "选择题" else "填空题"

    extraction_type = _normalize_to_binary_type(prompt_type)
    return prompt_type, extraction_type

# 文本模型推理
def inference_text_model(items, model_path):
    """
    使用文本模型对无图表题目进行批量推理
    
    Args:
        items: 数据列表
        model_path: 文本模型路径
    
    Returns:
        list: 添加了推理结果的数据列表
    """
    if not items:
        return items
    
    # 构建批量prompt
    prompts_list = []
    for item in items:
        ocr_text = item['ocr_result']['text']
        prompt_type, extraction_type = resolve_question_type(item)
        # 将解析后的题型写回，便于排查
        item['question_type'] = prompt_type
        
        # 获取文本模型提示词（按题型）
        prompt_template = get_text_prompt(prompt_type)
        system_prompt = prompt_template['system']
        user_prompt = prompt_template['user']
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\n{ocr_text}"}
        ]
        prompts_list.append(messages)
    
    # 批量推理
    print(f"文本模型批量推理中...")
    answers = text.generate_answers(model_path, prompts_list, max_tokens=MAX_TOKENS_TEXT)
    
    # 将结果添加到items中
    for i, item in enumerate(items):
        item['model_output'] = answers[i]
        
        # 提取答案
        _, extraction_type = resolve_question_type(item)
        extracted_answer, status = answer_extractor.extract_answer(answers[i], extraction_type)
        
        # 如果提取失败,使用默认答案
        if extracted_answer is None or status != "成功":
            extracted_answer = get_default_answer(extraction_type)
            status = "使用默认答案"
        
        item['answer'] = extracted_answer  # 改为answer字段,与视觉模型保持一致
        item['extraction_status'] = status
    
    return items

# 视觉模型推理
def inference_vision_model(items, image_dir, model_path):
    """
    使用视觉模型对有图表题目进行批量推理
    
    Args:
        items: 数据列表
        image_dir: 图像目录
        model_path: 视觉模型路径
    
    Returns:
        list: 添加了推理结果的数据列表
    """
    if not items:
        return items
    
    # 构建批量输入 (image_path, question)
    inputs_list = []
    for item in items:
        image_path = os.path.join(image_dir, item['image'])
        ocr_text = item['ocr_result']['text']
        prompt_type, extraction_type = resolve_question_type(item)
        # 将解析后的题型写回，便于排查
        item['question_type'] = prompt_type
        
        # 获取视觉模型提示词（按题型）
        prompt_template = get_vl_prompt(prompt_type)
        system_prompt = prompt_template['system']
        user_prompt = prompt_template['user']
        
        # 构建问题文本
        question = f"{system_prompt}\n\n{user_prompt}".strip()
        
        inputs_list.append((image_path, question))
    
    # 批量推理
    print(f"视觉模型批量推理中...")
    answers = vl.generate_answers(model_path, inputs_list, max_tokens=MAX_TOKENS_VL)
    
    # 将结果添加到items中
    for i, item in enumerate(items):
        item['model_output'] = answers[i]
        
        # 提取答案
        _, extraction_type = resolve_question_type(item)
        extracted_answer, status = answer_extractor.extract_answer(answers[i], extraction_type)
        
        # 如果提取失败,使用默认答案
        if extracted_answer is None or status != "成功":
            extracted_answer = get_default_answer(extraction_type)
            status = "使用默认答案"
        
        item['answer'] = extracted_answer
        item['extraction_status'] = status
    
    return items

# ==================== 阶段1: OCR识别 ====================
def stage1_ocr(image_dir, input_jsonl, ocr_output_jsonl):
    """阶段1: OCR识别并保存结果（使用 vLLM 批量推理）"""
    print("=" * 80)
    print("【阶段1: OCR识别（vLLM 批量推理）】")
    print("=" * 80)
    
    input_data = load_jsonl(input_jsonl)
    print(f"加载了 {len(input_data)} 条数据")
    
    # 加载OCR模型
    ocr_model = OCRModel(model_id=OCR_MODEL_PATH)
    
    # 收集所有图像路径
    print("\n收集所有图像路径...")
    all_image_paths = [os.path.join(image_dir, item['image']) for item in input_data]
    
    # 一次性批量OCR推理（vLLM自动进行动态批处理）
    print(f"开始批量OCR推理（共 {len(all_image_paths)} 张图像）...")
    all_ocr_results = ocr_model.ocr_batch(all_image_paths)
    
    # 将结果添加到对应的item中
    results = []
    for item, ocr_result in zip(input_data, all_ocr_results):
        item['ocr_result'] = ocr_result
        results.append(item)
    
    # 彻底释放OCR模型显存
    print("\n彻底释放OCR模型显存...")
    import time
    try:
        ocr_model.release()
        del ocr_model
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"显存使用: {allocated:.2f}GB (已分配) / {reserved:.2f}GB (已保留)")
        
        # 等待确保释放完成
        time.sleep(3)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("OCR模型显存释放完成")
    except Exception as e:
        print(f"释放OCR模型资源时出错: {e}")
    
    # 保存OCR结果
    save_jsonl(results, ocr_output_jsonl)
    print(f"\nOCR结果已保存到: {ocr_output_jsonl}")
    print(f"共处理 {len(results)} 条数据")
    
    # 统计图表检测结果
    no_chart_count = sum(1 for item in results if not item.get('ocr_result', {}).get('is_chart', False))
    chart_count = sum(1 for item in results if item.get('ocr_result', {}).get('is_chart', False))
    print(f"无图表题目: {no_chart_count} 条")
    print(f"有图表题目: {chart_count} 条")


# ==================== 阶段2: 文本模型推理 ====================
def stage2_text_inference(image_dir, ocr_jsonl, text_output_jsonl):
    """阶段2: 文本模型推理（无图表题目）"""
    print("=" * 80)
    print("【阶段2: 文本模型推理】")
    print("=" * 80)
    
    # 加载OCR结果
    all_data = load_jsonl(ocr_jsonl)
    print(f"加载了 {len(all_data)} 条OCR结果")
    
    # 筛选无图表题目
    no_chart_items = [item for item in all_data 
                     if not item.get('ocr_result', {}).get('is_chart', False)]
    
    print(f"无图表题目: {len(no_chart_items)} 条")
    
    if no_chart_items:
        print(f"\n开始文本模型推理...")
        no_chart_items = inference_text_model(no_chart_items, TEXT_MODEL_PATH)
        
        # 释放文本模型显存
        print("\n释放文本模型显存...")
        import time
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            gc.collect()
            time.sleep(3)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("文本模型显存释放完成")
        except Exception as e:
            print(f"释放文本模型资源时出错: {e}")
        
        # 保存文本模型推理结果
        save_jsonl(no_chart_items, text_output_jsonl)
        print(f"\n文本模型结果已保存到: {text_output_jsonl}")
        print(f"共处理 {len(no_chart_items)} 条数据")
    else:
        print("没有无图表题目，跳过文本模型推理")
        save_jsonl([], text_output_jsonl)


# ==================== 阶段3: 视觉模型推理 ====================
def stage3_vision_inference(image_dir, ocr_jsonl, text_jsonl, final_output_jsonl):
    """阶段3: 视觉模型推理（有图表题目）并合并最终结果"""
    print("=" * 80)
    print("【阶段3: 视觉模型推理与结果合并】")
    print("=" * 80)
    
    # 加载OCR结果
    all_data = load_jsonl(ocr_jsonl)
    print(f"加载了 {len(all_data)} 条OCR结果")
    
    # 加载文本模型结果
    text_results = load_jsonl(text_jsonl)
    print(f"加载了 {len(text_results)} 条文本模型结果")
    
    # 创建文本结果字典（按id索引）
    text_dict = {item.get('id', item.get('image')): item for item in text_results}
    
    # 筛选有图表题目
    chart_items = [item for item in all_data 
                  if item.get('ocr_result', {}).get('is_chart', False)]
    
    print(f"有图表题目: {len(chart_items)} 条")
    
    if chart_items:
        print(f"\n开始视觉模型推理...")
        chart_items = inference_vision_model(chart_items, image_dir, VL_MODEL_PATH)
        
        # 释放视觉模型显存
        print("\n释放视觉模型显存...")
        import time
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            gc.collect()
            time.sleep(3)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("视觉模型显存释放完成")
        except Exception as e:
            print(f"释放视觉模型资源时出错: {e}")
    else:
        print("没有有图表题目，跳过视觉模型推理")
    
    # 合并结果
    print("\n合并推理结果...")
    final_results = []
    
    for item in all_data:
        item_id = item.get('id', item.get('image'))
        
        # 优先使用文本模型结果
        if item_id in text_dict:
            final_results.append(text_dict[item_id])
        # 否则查找视觉模型结果
        else:
            chart_item = next((ci for ci in chart_items 
                             if ci.get('id', ci.get('image')) == item_id), None)
            if chart_item:
                final_results.append(chart_item)
            else:
                # 兜底：保留原始OCR结果
                final_results.append(item)
    
    # 按原始顺序排序
    if final_results and 'id' in final_results[0]:
        final_results.sort(key=lambda x: x.get('id', 0))
    
    # 保存最终结果
    save_jsonl(final_results, final_output_jsonl)
    print(f"\n最终结果已保存到: {final_output_jsonl}")
    print(f"共处理 {len(final_results)} 条数据")


# 主推理函数（完整流程，保持向后兼容）
def main(image_dir, input_jsonl, output_jsonl):
    """完整流程运行（不分阶段）"""
    import tempfile
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as ocr_temp:
        ocr_temp_path = ocr_temp.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as text_temp:
        text_temp_path = text_temp.name
    
    try:
        # 执行三个阶段
        stage1_ocr(image_dir, input_jsonl, ocr_temp_path)
        stage2_text_inference(image_dir, ocr_temp_path, text_temp_path)
        stage3_vision_inference(image_dir, ocr_temp_path, text_temp_path, output_jsonl)
    finally:
        # 清理临时文件
        import os
        try:
            os.remove(ocr_temp_path)
            os.remove(text_temp_path)
        except:
            pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  完整流程: python run.py <image_dir> <input_jsonl> <output_jsonl>")
        print("  阶段1(OCR): python run.py stage1 <image_dir> <input_jsonl> <ocr_output_jsonl>")
        print("  阶段2(文本): python run.py stage2 <image_dir> <ocr_jsonl> <text_output_jsonl>")
        print("  阶段3(视觉): python run.py stage3 <image_dir> <ocr_jsonl> <text_jsonl> <final_output_jsonl>")
        sys.exit(1)
    
    # 分阶段运行
    if sys.argv[1] == "stage1":
        stage1_ocr(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "stage2":
        stage2_text_inference(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "stage3":
        stage3_vision_inference(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    # 完整流程运行
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])