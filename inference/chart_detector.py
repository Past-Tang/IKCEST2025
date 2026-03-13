"""
图表检测模块 - 基于 OCR 文本关键词判断题目是否含图表

置信度评分规则：
    图表关键词（图/坐标/函数图像…）    +0.3
    图形引用（如图所示/由图可知…）      +0.3
    测量术语（面积/体积/半径…）         +0.2
    几何图形（三角形/长方体…）          +0.15
    结构特征（括号对/特殊符号…）        +0.1
    阈值 >= 0.3 判定为有图表
"""

import re
from typing import List, Dict


class MathProblemChartDetector:

    def __init__(self):
        self.chart_keywords = [
            '图', '图表', '图示', '图像', '图形', '坐标', '坐标系', '坐标轴',
            '横轴', '纵轴', 'x轴', 'y轴', '原点', '函数图像', '曲线',
            '直线', '抛物线', '椭圆', '双曲线', '扇形', '圆形', '矩形',
            '三角形', '长方体', '正方体', '圆柱', '圆锥', '球体',
            '如图所示', '如图示', '如右图', '如左图', '如上图', '如下图所示'
        ]
        self.measurement_keywords = [
            '边长', '周长', '面积', '体积', '半径', '直径', '圆心', '角度',
            '弧度', '高', '宽', '长', '对角线', '表面积', '容积'
        ]

    def detect_chart_in_math_problem(self, ocr_text: str, simple_mode: bool = True):
        cleaned = re.sub(r'\s+', ' ', ocr_text.strip()) if ocr_text else ""

        indicators = {
            'has_chart_keywords': any(kw in cleaned for kw in self.chart_keywords),
            'has_measurement_terms': any(kw in cleaned for kw in self.measurement_keywords),
            'has_geometric_shapes': bool(re.search(
                r'[长短]方形|[正长]方体|圆[柱锥形]|三[角棱]形|平行[四边形]|梯形|棱[柱锥台]', cleaned)),
            'has_figure_references': bool(re.search(
                r'如图\S*所示|参见图\S*|图\S*中|由图\S*可知', cleaned)),
            'text_structure_features': {
                'bracket_pairs': len(re.findall(r'\([^)]*\)', cleaned)),
                'alphanumeric_combos': len(re.findall(r'[A-Za-z]\d+', cleaned)),
                'special_chars': len(re.findall(r'[°∠△□○◇☆◎]', cleaned)),
            }
        }

        score = 0.0
        if indicators['has_chart_keywords']:     score += 0.3
        if indicators['has_measurement_terms']:  score += 0.2
        if indicators['has_geometric_shapes']:   score += 0.15
        if indicators['has_figure_references']:  score += 0.3
        s = indicators['text_structure_features']
        if s['bracket_pairs'] >= 2:              score += 0.1
        if s['alphanumeric_combos'] >= 2:        score += 0.1
        if s['special_chars'] >= 1:              score += 0.1
        score = min(score, 1.0)

        has_chart = score >= 0.3

        if simple_mode:
            return has_chart

        return {
            'has_chart': has_chart,
            'confidence_score': round(score, 3),
            'indicators': indicators,
        }


def batch_detect_charts(ocr_texts: List[str], simple_mode: bool = True) -> List:
    detector = MathProblemChartDetector()
    results = []
    for text in ocr_texts:
        if text and text.strip():
            results.append(detector.detect_chart_in_math_problem(text, simple_mode))
        else:
            results.append(False if simple_mode else {'has_chart': False, 'confidence_score': 0.0})
    return results
