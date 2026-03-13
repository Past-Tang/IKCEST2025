#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore", message=".*truncation.*")

from vllm import LLM, SamplingParams
from PIL import Image
import torch
from chart_detector import MathProblemChartDetector
from prompt import get_ocr_prompt
import gc

# ==================== 全局配置 ====================
# 图像尺寸：所有输入图像将被调整为此尺寸
IMAGE_SIZE = (720, 720)

class OCRModel:
    def __init__(self, model_id="./Ovis2-2B", enable_chart_detection=True):
        """初始化OCR模型
        
        Args:
            model_id: 模型ID或路径
            enable_chart_detection: 是否启用图表检测
        """
        print(f"正在加载OCR模型（Ovis2-2B + vLLM）: {model_id}")
        
        # 使用 vLLM 加载模型
        # 针对 V100-32GB 优化配置
        self.llm = LLM(
                model=model_id,
                max_model_len=8192,
                max_num_seqs=8,  # 降低并发数
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1, "video": 0},  # 禁用视频，只用图像
            )
        
        # 获取 tokenizer（从模型中）
        self.tokenizer = self.llm.get_tokenizer()
        
        print(f"OCR模型加载完成（vLLM）")
        
        # 初始化图表检测器
        self.enable_chart_detection = enable_chart_detection
        if self.enable_chart_detection:
            self.chart_detector = MathProblemChartDetector()
            print("图表检测器已启用")
     
    def ocr_batch(self, image_paths, detect_chart=None):
        """批量OCR识别（利用 vLLM 批量推理能力）
        
        Args:
            image_paths: 图像文件路径列表
            detect_chart: 是否检测图表，None时使用初始化设置
            
        Returns:
            list[dict]: 每个元素包含OCR文本和图表检测结果
                [
                    {
                        'text': str,  # OCR识别的文本
                        'is_chart': bool or None  # 是否为图表题
                    },
                    ...
                ]
        """
        if not image_paths:
            return []
        
        try:
            from PIL import Image as PILImage
            
            # 构建批量输入（从 prompt.py 获取提示词）
            batch_inputs = []
            query = get_ocr_prompt()
            
            # 为所有图像构建消息
            all_messages = [
                [{"role": "user", "content": f"<image>\n{query}"}]
                for _ in image_paths
            ]
            
            # 批量生成提示词
            prompts = self.tokenizer.apply_chat_template(
                all_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 加载图像并构建批量输入
            for idx, image_path in enumerate(image_paths):
                image = PILImage.open(image_path).convert("RGB")
                # 调整图像尺寸
                image = image.resize(IMAGE_SIZE)
                batch_inputs.append({
                    "prompt": prompts[idx],
                    "multi_modal_data": {"image": image},
                })
            
            # 设置采样参数（保持不变）
            sampling_params = SamplingParams(
                temperature=0.4,
                max_tokens=2048,
                repetition_penalty=1.05,   # 重复惩罚，大于1表示降低重复
                frequency_penalty=0.1,    # 频率惩罚，大于0表示降低频繁出现的token的概率
                presence_penalty=0.1,     # 存在惩罚，大于0表示降低已经出现过的token的概率
            )
            
            # ==================== 修改部分：分成4批进行推理 ====================
            total_batches = len(batch_inputs)
            batch_size = total_batches // 4
            remainder = total_batches % 4
            
            all_outputs = []
            
            # 处理前3个批次（大小相同）
            for i in range(3):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                print(f"开始第{i+1}批次推理: {batch_size} 个样本")
                current_batch = batch_inputs[start_idx:end_idx]
                current_outputs = self.llm.generate(current_batch, sampling_params=sampling_params)
                all_outputs.extend(current_outputs)
                
                # 清理当前批次的内存
                del current_batch, current_outputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 处理第4个批次（包含剩余的所有样本）
            start_idx = 3 * batch_size
            last_batch_size = batch_size + remainder
            
            print(f"开始第4批次推理: {last_batch_size} 个样本")
            last_batch = batch_inputs[start_idx:]
            last_outputs = self.llm.generate(last_batch, sampling_params=sampling_params)
            all_outputs.extend(last_outputs)
            
            # 清理最后批次的内存
            del last_batch, last_outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs = all_outputs
            # ==================== 修改部分结束 ====================
            
            # 处理结果
            results = []
            should_detect = detect_chart if detect_chart is not None else self.enable_chart_detection
            
            for output in outputs:
                text = output.outputs[0].text.strip()
                result = {
                    'text': text,
                    'is_chart': None
                }
                
                # 图表检测
                if should_detect and text:
                    is_chart = self.chart_detector.detect_chart_in_math_problem(text, simple_mode=True)
                    result['is_chart'] = is_chart
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"批量OCR识别失败: {str(e)}")
            # 返回空结果列表
            return [{'text': "", 'is_chart': None} for _ in image_paths]
    
    def ocr_image_simple(self, image_path):
        """简化版OCR识别，仅返回文本（保持向后兼容）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: OCR识别结果字符串
        """
        result = self.ocr_image(image_path, detect_chart=False)
        return result['text']

    def release(self):
        """释放模型资源"""
        # 删除图表检测器
        if hasattr(self, 'chart_detector'):
            del self.chart_detector
            self.chart_detector = None
        
        # 删除 tokenizer
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        
        # 删除 LLM
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
        
        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 测试代码
if __name__ == "__main__":
    # 测试OCR与图表检测
    print("=" * 60)
    print("OCR + 图表检测测试")
    print("=" * 60)