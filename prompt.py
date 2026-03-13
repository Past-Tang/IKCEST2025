#!/usr/bin/env python3
"""
提示词模块 - 为文本模型和视觉模型提供不同的提示词
"""

# ==================== OCR 提示词 ====================
OCR_PROMPT = '''输出图像中的完整题目 不要解答！'''

# ==================== 数学题求解提示词 ====================
MATH_SOLVE_BASE = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}."

MATH_SOLVE_CHOICE = """Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.
For multiple choice questions, provide ONLY the option letter (A, B, C, or D).
Format: \\boxed{A}"""


# ==================== 视觉数学题求解提示词 ====================
MATH_SOLVE_BASE = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}."

MATH_SOLVE_CHOICE = """Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.
For multiple choice questions, provide ONLY the option letter (A, B, C, or D).
Format: \\boxed{A}"""




# ==================== 文本模型提示词 ====================
# 文本模型 - 选择题提示词
TEXT_CHOICE_PROMPT = {
    "system": "",
    "user": MATH_SOLVE_CHOICE
}

# 文本模型 - 非选择题提示词（填空题、计算应用题）
TEXT_NON_CHOICE_PROMPT = {
    "system": "",
    "user": MATH_SOLVE_BASE
}


# ==================== 视觉模型提示词 ====================
# 视觉模型 - 选择题提示词
VL_CHOICE_PROMPT = {
    "system": """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""",
    "user": MATH_SOLVE_CHOICE
}

# 视觉模型 - 非选择题提示词（填空题、计算应用题）
VL_NON_CHOICE_PROMPT = {
    "system": """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""",
    "user": MATH_SOLVE_BASE
}

# ==================== 题型映射 ====================
# 文本模型提示词映射
TEXT_PROMPTS = {
    "选择题": TEXT_CHOICE_PROMPT,
    "填空题": TEXT_NON_CHOICE_PROMPT,
    "计算应用题": TEXT_NON_CHOICE_PROMPT
}

# 视觉模型提示词映射
VL_PROMPTS = {
    "选择题": VL_CHOICE_PROMPT,
    "填空题": VL_NON_CHOICE_PROMPT,
    "计算应用题": VL_NON_CHOICE_PROMPT
}


def get_prompt(question_type, model_type="text"):
    """
    根据题型和模型类型获取提示词
    
    参数:
        question_type: 题型，如 "选择题"、"填空题"、"计算应用题"
        model_type: 模型类型，"text"（文本模型）或 "vl"（视觉模型），默认为 "text"
    
    返回:
        dict: {"system": 系统提示词, "user": 用户提示词}
    """
    if model_type == "vl":
        return VL_PROMPTS.get(question_type, VL_PROMPTS["填空题"])
    else:
        return TEXT_PROMPTS.get(question_type, TEXT_PROMPTS["填空题"])


def get_text_prompt(question_type):
    """
    获取文本模型提示词（便捷函数）
    
    参数:
        question_type: 题型，如 "选择题"、"填空题"、"计算应用题"
    
    返回:
        dict: {"system": 系统提示词, "user": 用户提示词}
    """
    return get_prompt(question_type, model_type="text")


def get_vl_prompt(question_type):
    """
    获取视觉模型提示词（便捷函数）
    
    参数:
        question_type: 题型，如 "选择题"、"填空题"、"计算应用题"
    
    返回:
        dict: {"system": 系统提示词, "user": 用户提示词}
    """
    return get_prompt(question_type, model_type="vl")


def get_ocr_prompt():
    """
    获取OCR提示词
    
    返回:
        str: OCR提示词
    """
    return OCR_PROMPT


if __name__ == "__main__":
    # 测试
    print("=" * 80)
    print("=== 提示词测试 ===")
    print("=" * 80)
    
    for qtype in ["选择题", "填空题", "计算应用题"]:
        print(f"\n【{qtype}】")
        
        # 文本模型提示词
        text_prompt = get_text_prompt(qtype)
        print(f"\n  文本模型系统提示词:")
        print(f"  {text_prompt['system'][:80]}...")
        
        # 视觉模型提示词
        vl_prompt = get_vl_prompt(qtype)
        print(f"\n  视觉模型系统提示词:")
        print(f"  {vl_prompt['system'][:80]}...")
        
        print("\n" + "-" * 80)
    
    print("\n测试完成！")
