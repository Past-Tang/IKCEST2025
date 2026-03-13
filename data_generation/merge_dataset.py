"""
SFT 数据集合并脚本

合并多个来源的数据（有图/无图），自动添加 <think> 格式提示词，
按题型分配对应的提示词模板。

用法：修改 main() 中的路径配置后直接运行
"""

import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm


CHOICE_PROMPT = {
    "system": "",
    "user": '''You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
        The reasoning process MUST BE enclosed within <think> </think> tags.
        The final answer MUST BE put in \\boxed{}.
        For multiple choice questions, provide ONLY the option letter (A, B, C, or D).
        Format: \\boxed{A}'''
}

NON_CHOICE_PROMPT = {
    "system": "",
    "user": '''You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
        The reasoning process MUST BE enclosed within <think> </think> tags.
        The final answer MUST BE put in \\boxed{}.'''
}


def get_prompt_by_tag(tag):
    return CHOICE_PROMPT if tag == "选择题" else NON_CHOICE_PROMPT


def convert_to_sft_text(item):
    """无图数据转 SFT 格式"""
    prompt = get_prompt_by_tag(item.get('tag', ''))
    question = item.get('question_text', '')
    answer = item.get('r1_solution_1', '')

    messages = []
    if prompt['system']:
        messages.append({"role": "system", "content": prompt['system']})
    messages.append({"role": "user", "content": f"{question}\n\n{prompt['user']}"})
    messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def convert_to_sft_image(item, image_dir="images", source="jsonl"):
    """有图数据转 SFT 格式"""
    prompt = get_prompt_by_tag(item.get('tag', ''))

    if source == "jsonl":
        image_path = item.get('image', '')
        answer = item.get('r1_solution_1', '')
        question = item.get('question_text', '')
    else:
        image_path = item.get('image', '')
        answer = item.get('model_response', '')
        question = ''

    new_image = f"{image_dir}/{Path(image_path).name}" if image_path else ''

    messages = []
    if prompt['system']:
        messages.append({"role": "system", "content": prompt['system']})

    user_content = f"<image> {question}\n\n{prompt['user']}" if question else f"<image>\n\n{prompt['user']}"
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": answer})

    result = {"messages": messages}
    if new_image:
        result["images"] = [new_image]

    return result, image_path, new_image


def main():
    """修改以下路径配置后运行"""
    random.seed(42)

    base_dir = Path("./source_data")
    output_dir = Path("./output_sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("请修改 main() 中的路径配置后运行")
    print("功能：合并有图/无图数据，自动添加提示词模板")


if __name__ == "__main__":
    main()
