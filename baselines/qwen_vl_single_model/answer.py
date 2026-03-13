"""
Baseline 答案提取 - 与主方案共用 MathAnswerExtractor

复制自 inference/answer.py，保持 baseline 独立可运行。
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
            matches = re.findall(r'\\boxed\{([A-Z])\}', model_answer)
            if matches:
                return matches[-1]

            extracted = parse(model_answer, extraction_config=[self.string_config])
            if extracted and len(extracted) > 0:
                result = str(extracted[0])
                if result and len(result) == 1 and result.isupper() and result.isalpha():
                    return result

            patterns = [
                r'答案[为是：:]\s*([A-Z])',
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
            if not extracted:
                return None
            answer = extracted[0]
            if answer is None:
                return None
            try:
                val = float(answer.evalf()) if hasattr(answer, 'evalf') else float(answer)
                return f"{val:.6f}"
            except (ValueError, TypeError, AttributeError):
                try:
                    return f"{float(str(answer)):.6f}"
                except ValueError:
                    return None
        except Exception:
            return None

    def extract_answer(self, model_answer: str, question_type: str) -> Tuple[Optional[str], str]:
        if question_type == "选择题":
            answer = self.extract_choice_answer(model_answer)
        else:
            answer = self.extract_numeric_answer(model_answer)
        return (answer, "成功") if answer else (None, "格式错误")
