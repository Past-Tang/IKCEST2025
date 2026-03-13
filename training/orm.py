"""
GRPO 自定义奖励函数 (Outcome Reward Model)

奖励规则：
    <think> 标签存在且非空       +0.3 / -0.8
    think 长度合理 (4000-9016 tokens) +0.3 / -0.2~-0.5
    \\boxed{} 格式存在           +0.2 / -0.5
    答案正确 (math_verify 验证)   +1.0 / -0.2
    总输出长度 < 11800 tokens    +0.15 / -0.8

使用方法：
    swift rlhf --external_plugins orm.py --reward_funcs custom_math_reward
"""

import re
from typing import List, Optional
from swift.plugin import ORM, orms


class CustomMathORM(ORM):

    def __init__(self):
        super().__init__()
        from math_verify import parse, verify
        self.parse = parse
        self.verify = verify

    def extract_last_boxed_answer(self, text: str) -> Optional[str]:
        """提取最后一个 \\boxed{} 中的内容"""
        matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        return matches[-1].strip() if matches else None

    def verify_answer_correctness(self, student_answer: str, ground_truth: str) -> bool:
        """使用 math_verify 验证答案正确性，失败时回退到字符串比较"""
        try:
            parsed_student = self.parse(student_answer)
            parsed_truth = self.parse(ground_truth)
            return float(self.verify(parsed_student, parsed_truth)) > 0
        except Exception:
            return student_answer.strip() == ground_truth.strip()

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for content, sol in zip(completions, solution):
            reward = 0.0

            # 1. <think></think> 标签检查
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                if think_content:
                    reward += 0.3
                    think_len = len(think_content)
                    if 8000 <= think_len <= 17999:
                        reward += 0.3
                    elif think_len < 4000:
                        reward -= 0.5
                    elif think_len < 8000:
                        reward -= 0.2
                    else:
                        reward -= 0.45
                else:
                    reward -= 0.8
            else:
                reward -= 0.8

            # 2. \\boxed{} 格式检查
            student_answer = self.extract_last_boxed_answer(content)
            if student_answer:
                reward += 0.2
            else:
                reward -= 0.5

            # 3. 答案正确性
            if student_answer:
                ground_truth = self.extract_last_boxed_answer(sol) or sol.strip()
                if self.verify_answer_correctness(student_answer, ground_truth):
                    reward += 1.0
                else:
                    reward -= 0.2

            # 4. 总输出长度控制
            if len(content) < 11800:
                reward += 0.15
            else:
                reward -= 0.8

            rewards.append(reward)

        return rewards


class StrictFormatORM(ORM):
    """严格格式检查：<think> 非空 + \\boxed{} → +1.0，否则 -1.0"""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            has_think = bool(re.search(r'<think>[\s\S]+?</think>', content))
            has_boxed = bool(re.search(r'\\boxed\{[^}]+\}', content))
            rewards.append(1.0 if (has_think and has_boxed) else -1.0)
        return rewards


orms['custom_math_reward'] = CustomMathORM
orms['strict_format'] = StrictFormatORM
