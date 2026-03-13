#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学答案提取脚本

功能：
    - 支持选择题答案提取：返回单个选项字母（如A、B、C、D）
    - 支持非选择题答案提取：返回保留六位小数的浮点数
    - 使用Math-Verify库进行数学表达式解析

使用示例：
    from extract_math_answers import MathAnswerExtractor
    
    extractor = MathAnswerExtractor()
    
    # 选择题
    answer, status = extractor.extract_answer("答案为 \\boxed{C}", "选择题")
    # 返回: ('C', '成功')
    
    # 填空题/计算应用题
    answer, status = extractor.extract_answer("答案为 \\boxed{3.14}", "填空题")
    # 返回: ('3.140000', '成功')
"""

import re
from typing import Optional, Tuple
from math_verify import parse
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig


class MathAnswerExtractor:
    
    def __init__(self):
        """
        初始化答案提取器
        
        配置三种提取模式，以适应不同类型的数学答案格式。
        """
        self.latex_config = LatexExtractionConfig()
        self.expr_config = ExprExtractionConfig()
        self.string_config = StringExtractionConfig()
    
    def extract_choice_answer(self, model_answer: str) -> Optional[str]:
        try:
            # 策略1：从\boxed{}中提取所有单个大写字母，返回最后一个
            boxed_pattern = r'\\boxed\{([A-Z])\}'
            boxed_matches = re.findall(boxed_pattern, model_answer)
            if boxed_matches:
                return boxed_matches[-1]  # 返回最后一个匹配的大写字母
            
            # 策略2：使用Math-Verify的字符串提取配置
            # 这会尝试从文本中提取字符串类型的答案
            extracted = parse(model_answer, extraction_config=[self.string_config])
            if extracted and len(extracted) > 0:
                result = str(extracted[0])
                # 验证结果是否为单个大写字母
                if result and len(result) == 1 and result.isupper() and result.isalpha():
                    return result
            
            # 策略3：使用多个正则表达式模式匹配常见的答案表述方式
            patterns = [
                r'答案[为是：:]\s*([A-Z])',           # 中文：答案为A、答案是B
                r'选项\s*[(\[]?\s*([A-Z])\s*[)\]]?',  # 中文：选项(A)、选项[B]
                r'\\boxed\{([A-Z])\}',                 # LaTeX格式：\boxed{C}
                r'^([A-Z])$',                          # 单独的字母：A
                r'[^A-Za-z]([A-Z])[^A-Za-z]*$',       # 文本末尾的大写字母
            ]
            
            # 遍历所有模式，找到匹配项
            for pattern in patterns:
                matches = re.findall(pattern, model_answer, re.MULTILINE)
                if matches:
                    return matches[-1]  # 返回最后一个匹配（通常是最终答案）
            
            return None
            
        except Exception:
            # 发生任何异常，返回None
            return None
    
    def extract_numeric_answer(self, model_answer: str) -> Optional[str]:
        try:
            # 使用Math-Verify提取数学表达式
            # 优先使用LaTeX配置，其次使用表达式配置
            extracted = parse(
                model_answer, 
                extraction_config=[self.latex_config, self.expr_config]
            )
            
            # 检查是否提取到结果
            if not extracted or len(extracted) == 0:
                return None
            
            # 获取第一个提取的答案
            # 通常Math-Verify会从\boxed{}中提取答案
            answer = extracted[0]
            
            if answer is None:
                return None
            
            # 尝试将答案转换为浮点数
            try:
                # 情况1：SymPy对象（如sqrt(2), Rational(1,3)）
                # 使用evalf()方法将符号表达式数值化
                if hasattr(answer, 'evalf'):
                    numeric_value = float(answer.evalf())
                # 情况2：已经是数值类型（int, float）
                else:
                    numeric_value = float(answer)
                
                # 格式化为保留六位小数的字符串
                return f"{numeric_value:.6f}"
                
            except (ValueError, TypeError, AttributeError):
                # 如果上述方法失败，尝试字符串处理
                answer_str = str(answer)
                
                try:
                    # 尝试直接转换字符串为浮点数
                    numeric_value = float(answer_str)
                    return f"{numeric_value:.6f}"
                    
                except ValueError:
                    # 特殊处理：分数形式（如"1/3"）
                    if '/' in answer_str:
                        # 清理字符串，移除空格
                        cleaned = answer_str.replace(' ', '')
                        parts = cleaned.split('/')
                        
                        # 确保是标准分数形式（分子/分母）
                        if len(parts) == 2:
                            try:
                                numerator = float(parts[0])
                                denominator = float(parts[1])
                                
                                # 计算分数值（避免除零）
                                if denominator != 0:
                                    numeric_value = numerator / denominator
                                    return f"{numeric_value:.6f}"
                            except ValueError:
                                pass
                    
                    # 无法转换为数值，返回None
                    return None
                    
        except Exception:
            # 发生任何异常，返回None
            return None
    
    def extract_answer(self, model_answer: str, question_type: str) -> Tuple[Optional[str], str]:
        # 选择题：返回单个字母（A、B、C、D等）
        if question_type == "选择题":
            answer = self.extract_choice_answer(model_answer)
            if answer:
                return answer, "成功"
            else:
                return None, "格式错误"
        
        # 非选择题：返回保留六位小数的数值
        else:
            answer = self.extract_numeric_answer(model_answer)
            if answer:
                return answer, "成功"
            else:
                return None, "格式错误"