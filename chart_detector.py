import re
from typing import List, Dict


class MathProblemChartDetector:
    """数学题图表检测器"""
    
    def __init__(self):
        # 图表相关关键词
        self.chart_keywords = [
            '图', '图表', '图示', '图像', '图形', '坐标', '坐标系', '坐标轴',
            '横轴', '纵轴', 'x轴', 'y轴', '原点', '函数图像', '曲线',
            '直线', '抛物线', '椭圆', '双曲线', '扇形', '圆形', '矩形',
            '三角形', '长方体', '正方体', '圆柱', '圆锥', '球体',
            '如图所示', '如图示', '如右图', '如左图', '如上图', '如下图所示'
        ]
        
        # 图形测量相关词汇
        self.measurement_keywords = [
            '边长', '周长', '面积', '体积', '半径', '直径', '圆心', '角度',
            '弧度', '高', '宽', '长', '对角线', '表面积', '容积'
        ]
        
        # 不再检测坐标点模式

    def detect_chart_in_math_problem(self, ocr_text: str, simple_mode: bool = True) -> bool:
        """
        检测数学题是否包含图表
        
        Args:
            ocr_text: OCR识别出的文本
            simple_mode: 简化模式，直接返回True/False（默认为True）
            
        Returns:
            bool: 是否为图表题（simple_mode=True时）
            Dict: 详细检测结果（simple_mode=False时）
        """
        # 预处理文本
        cleaned_text = self._preprocess_text(ocr_text)
        
        # 各项检测指标（已去掉坐标点检测）
        indicators = {
            'has_chart_keywords': self._check_chart_keywords(cleaned_text),
            'has_measurement_terms': self._check_measurement_terms(cleaned_text),
            # 不再检测坐标点
            'has_geometric_shapes': self._check_geometric_shapes(cleaned_text),
            'has_figure_references': self._check_figure_references(cleaned_text),
            'text_structure_features': self._analyze_text_structure(cleaned_text)
        }
        
        # 综合评分
        confidence_score = self._calculate_confidence_score(indicators)
        
        # 最终判断
        has_chart = confidence_score >= 0.3  # 阈值可根据实际情况调整
        
        # 简化模式：直接返回布尔值
        if simple_mode:
            return has_chart
        
        # 详细模式：返回完整信息
        return {
            'has_chart': has_chart,
            'confidence_score': round(confidence_score, 3),
            'indicators': indicators,
            'reasoning': self._generate_reasoning(indicators, has_chart)
        }
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""
        # 去除多余空格和换行符
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _check_chart_keywords(self, text: str) -> bool:
        """检查是否包含图表关键词"""
        for keyword in self.chart_keywords:
            if keyword in text:
                return True
        return False
    
    def _check_measurement_terms(self, text: str) -> bool:
        """检查是否包含测量相关词汇"""
        for keyword in self.measurement_keywords:
            if keyword in text:
                return True
        return False
    
    # 移除 _check_coordinate_points 方法
    # def _check_coordinate_points(self, text: str) -> bool:
    #     ...

    def _check_geometric_shapes(self, text: str) -> bool:
        """检查几何图形相关描述"""
        shape_patterns = [
            r'[长短]方形', r'[正长]方体', r'圆[柱锥形]', r'三[角棱]形', 
            r'平行[四边形]', r'梯形', r'棱[柱锥台]'
        ]
        
        for pattern in shape_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _check_figure_references(self, text: str) -> bool:
        """检查图表引用"""
        figure_ref_patterns = [
            r'如图\S*所示', r'参见图\S*', r'图\S*中', r'由图\S*可知'
        ]
        
        for pattern in figure_ref_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _analyze_text_structure(self, text: str) -> Dict:
        """分析文本结构特征"""
        # 计算各种标点符号和特殊字符的比例
        features = {}
        
        # 括号对数量（可能表示坐标点）
        bracket_pairs = len(re.findall(r'\([^)]*\)', text))
        features['bracket_pairs'] = bracket_pairs
        
        # 字母数字组合（可能表示点标记）
        alphanumeric_combos = len(re.findall(r'[A-Za-z]\d+', text))
        features['alphanumeric_combos'] = alphanumeric_combos
        
        # 特殊符号
        special_chars = len(re.findall(r'[°∠△□○◇☆◎]', text))
        features['special_chars'] = special_chars
        
        return features
    
    def _calculate_confidence_score(self, indicators: Dict) -> float:
        """计算置信度分数"""
        score = 0.0
        
        # 关键词检测权重
        if indicators['has_chart_keywords']:
            score += 0.3
        if indicators['has_measurement_terms']:
            score += 0.2
        # 不再参与坐标点加分
        # if indicators['has_coordinate_points']:
        #     score += 0.25
        if indicators['has_geometric_shapes']:
            score += 0.15
        if indicators['has_figure_references']:
            score += 0.3
            
        # 结构特征权重
        structure = indicators['text_structure_features']
        if structure['bracket_pairs'] >= 2:
            score += 0.1
        if structure['alphanumeric_combos'] >= 2:
            score += 0.1
        if structure['special_chars'] >= 1:
            score += 0.1
            
        return min(score, 1.0)  # 确保不超过1.0
    
    def _generate_reasoning(self, indicators: Dict, has_chart: bool) -> str:
        """生成判断理由"""
        reasons = []
        
        if indicators['has_chart_keywords']:
            reasons.append("检测到图表相关关键词")
        if indicators['has_measurement_terms']:
            reasons.append("包含几何测量术语")
        # 不再添加“发现坐标点表示”的理由
        # if indicators.get('has_coordinate_points', False):
        #     reasons.append("发现坐标点表示")
        if indicators['has_geometric_shapes']:
            reasons.append("提及几何图形")
        if indicators['has_figure_references']:
            reasons.append("有明确的图表引用")
            
        if not reasons:
            reasons.append("未发现明显的图表特征")
            
        result = "带图表" if has_chart else "不带图表"
        return f"判断为{result}，原因：{'；'.join(reasons)}"


def batch_detect_charts(ocr_texts: List[str], simple_mode: bool = True) -> List:
    """
    批量检测多个OCR文本
    
    Args:
        ocr_texts: OCR识别文本列表
        simple_mode: 简化模式，直接返回True/False列表
        
    Returns:
        List[bool] 或 List[Dict]: 每个文本的检测结果
    """
    detector = MathProblemChartDetector()
    results = []
    
    for text in ocr_texts:
        if text and text.strip():  # 跳过空文本
            result = detector.detect_chart_in_math_problem(text, simple_mode=simple_mode)
            results.append(result)
        else:
            results.append(False if simple_mode else {
                'has_chart': False,
                'confidence_score': 0.0,
                'indicators': {},
                'reasoning': '输入文本为空'
            })
    
    return results


# 测试函数
def test_detector():
    """测试检测器"""
    detector = MathProblemChartDetector()
    
    # 测试用例
    test_cases = [
        # 带图表的题目
        "如图，在直角坐标系中，点A(2,3)，点B(5,1)，求线段AB的长度",
        "已知函数y=x²的图像如图所示，求该函数在x=2处的切线斜率",
        "在△ABC中，AB=5cm，BC=6cm，∠ABC=60°，求三角形面积",
        "一个圆柱的底面半径为3cm，高为8cm，求其体积",
        
        # 不带图表的题目
        "解方程：2x + 5 = 13",
        "计算：∫(0 to 1) x² dx",
        "证明：对于任意实数x，有x² ≥ 0",
        "已知数列{an}满足a1=1，an+1=2an+1，求通项公式"
    ]
    
    print("数学题图表检测测试结果：")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        # 简化模式
        is_chart = detector.detect_chart_in_math_problem(text, simple_mode=True)
        print(f"\n测试用例 {i}:")
        print(f"题目: {text}")
        print(f"判断结果: {'带图表' if is_chart else '不带图表'}")
        
        # 详细模式（用于调试）
        result = detector.detect_chart_in_math_problem(text, simple_mode=False)
        print(f"置信度: {result['confidence_score']}")
        print(f"理由: {result['reasoning']}")
        print("-" * 30)


if __name__ == "__main__":
    # 运行测试
    test_detector()
    
    # 示例：处理单个题目
    print("\n" + "=" * 50)
    detector = MathProblemChartDetector()
    sample_text = "在平面直角坐标系中，点P(3,4)到原点的距离是多少？"
    
    # 简化模式
    is_chart = detector.detect_chart_in_math_problem(sample_text, simple_mode=True)
    print(f"\n单题检测示例（简化模式）:")
    print(f"题目: {sample_text}")
    print(f"是否为图表题: {is_chart}")
    
    # 详细模式
    result = detector.detect_chart_in_math_problem(sample_text, simple_mode=False)
    print(f"\n单题检测示例（详细模式）:")
    print(f"完整结果: {result}")

