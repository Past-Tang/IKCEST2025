"""
Baseline 提示词 - Qwen2.5-VL-3B 单模型方案
"""

CHOICE_PROMPT = {
    "system": (
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The final answer MUST BE put in \\boxed{}. "
        "For multiple choice questions, provide ONLY the option letter (A, B, C, or D). "
        "Format: \\boxed{A}"
    ),
    "user": "Please think carefully and solve the problem shown in the image."
}

NON_CHOICE_PROMPT = {
    "system": (
        "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The final answer MUST BE put in \\boxed{}."
    ),
    "user": "Please think carefully and solve the problem shown in the image."
}

PROMPTS = {"选择题": CHOICE_PROMPT, "填空题": NON_CHOICE_PROMPT, "计算应用题": NON_CHOICE_PROMPT}


def get_prompt(question_type):
    return PROMPTS.get(question_type, PROMPTS["填空题"])
