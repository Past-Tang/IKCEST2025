"""
答案提取模块 - 从模型输出中提取标准化答案

策略：
    选择题：\\boxed{A} → 单个字母，多重正则兜底
    数值题：\\boxed{expr} → math_verify 解析 → SymPy evalf → 6位小数

示例：
    extractor = MathAnswerExtractor()
    extractor.extract_answer("\\boxed{C}", "选择题")   # ('C', '成功')
    extractor.extract_answer("\\boxed{3.14}", "填空题") # ('3.140000', '成功')
"""

import re
from typing import Optional, Tuple
from math_verify import parse
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig


class MathAnswerExtractor:

    def __init__(self):
        self.latex_config = LatexExtractionConfig()
        self.expr_config = ExprExtractionConfig()
        self.string_config = StringExtractionConfig()

    def extract_choice_answer(self, model_answer: str) -> Optional[str]:
        try:
            # 策略1：\boxed{} 中的大写字母（取最后一个）
            matches = re.findall(r'\\boxed\{([A-Z])\}', model_answer)
            if matches:
                return matches[-1]

            # 策略2：math_verify 字符串提取
            extracted = parse(model_answer, extraction_config=[self.string_config])
            if extracted and len(extracted) > 0:
                result = str(extracted[0])
                if result and len(result) == 1 and result.isupper() and result.isalpha():
                    return result

            # 策略3：多模式正则匹配
            patterns = [
                r'答案[为是：:]\s*([A-Z])',
                r'选项\s*[(\[]?\s*([A-Z])\s*[)\]]?',
                r'\\boxed\{([A-Z])\}',
                r'^([A-Z])$',
                r'[^A-Za-z]([A-Z])[^A-Za-z]*$',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, model_answer, re.MULTILINE)
                if matches:
                    return matches[-1]

            return None
        except Exception:
            return None

    def extract_numeric_answer(self, model_answer: str) -> Optional[str]:
        try:
            extracted = parse(model_answer, extraction_config=[self.latex_config, self.expr_config])
            if not extracted or len(extracted) == 0:
                return None

            answer = extracted[0]
            if answer is None:
                return None

            try:
                if hasattr(answer, 'evalf'):
                    numeric_value = float(answer.evalf())
                else:
                    numeric_value = float(answer)
                return f"{numeric_value:.6f}"
            except (ValueError, TypeError, AttributeError):
                answer_str = str(answer)
                try:
                    return f"{float(answer_str):.6f}"
                except ValueError:
                    if '/' in answer_str:
                        parts = answer_str.replace(' ', '').split('/')
                        if len(parts) == 2:
                            try:
                                num, den = float(parts[0]), float(parts[1])
                                if den != 0:
                                    return f"{num / den:.6f}"
                            except ValueError:
                                pass
                    return None
        except Exception:
            return None

    def extract_answer(self, model_answer: str, question_type: str) -> Tuple[Optional[str], str]:
        if question_type == "选择题":
            answer = self.extract_choice_answer(model_answer)
        else:
            answer = self.extract_numeric_answer(model_answer)
        return (answer, "成功") if answer else (None, "格式错误")
