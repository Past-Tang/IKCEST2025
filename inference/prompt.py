"""
提示词模块 - 按题型和模型类型分发提示词

OCR 阶段：只提取题目文本，不解答
文本模型：直接 \boxed{} 出答案
视觉模型：<think></think> 推理后再 \boxed{} 出答案
"""

# ==================== OCR 提示词 ====================
OCR_PROMPT = '''输出图像中的完整题目 不要解答！'''

# ==================== 数学求解提示词 ====================
MATH_SOLVE_BASE = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}."

MATH_SOLVE_CHOICE = """Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.
For multiple choice questions, provide ONLY the option letter (A, B, C, or D).
Format: \\boxed{A}"""


# ==================== 文本模型提示词 ====================
TEXT_CHOICE_PROMPT = {"system": "", "user": MATH_SOLVE_CHOICE}
TEXT_NON_CHOICE_PROMPT = {"system": "", "user": MATH_SOLVE_BASE}


# ==================== 视觉模型提示词（带 CoT 推理） ====================
_VL_SYSTEM = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
"""

VL_CHOICE_PROMPT = {"system": _VL_SYSTEM, "user": MATH_SOLVE_CHOICE}
VL_NON_CHOICE_PROMPT = {"system": _VL_SYSTEM, "user": MATH_SOLVE_BASE}


# ==================== 题型映射 ====================
TEXT_PROMPTS = {"选择题": TEXT_CHOICE_PROMPT, "填空题": TEXT_NON_CHOICE_PROMPT, "计算应用题": TEXT_NON_CHOICE_PROMPT}
VL_PROMPTS = {"选择题": VL_CHOICE_PROMPT, "填空题": VL_NON_CHOICE_PROMPT, "计算应用题": VL_NON_CHOICE_PROMPT}


def get_prompt(question_type, model_type="text"):
    prompts = VL_PROMPTS if model_type == "vl" else TEXT_PROMPTS
    return prompts.get(question_type, prompts["填空题"])

def get_text_prompt(question_type):
    return get_prompt(question_type, "text")

def get_vl_prompt(question_type):
    return get_prompt(question_type, "vl")

def get_ocr_prompt():
    return OCR_PROMPT
