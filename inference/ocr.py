#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 模块 - 使用 InternVL3.5-2B + vLLM 进行批量 OCR 识别

功能：
    - 批量图像 OCR 文本提取
    - 集成图表检测（判断题目是否含图表）
    - 分批推理避免 V100 OOM
"""

import os
import warnings
warnings.filterwarnings("ignore", message=".*truncation.*")

from vllm import LLM, SamplingParams
from PIL import Image
import torch
from chart_detector import MathProblemChartDetector
from prompt import get_ocr_prompt
import gc

IMAGE_SIZE = (720, 720)


class OCRModel:
    def __init__(self, model_id="./InternVL3_5-2B-Instruct", enable_chart_detection=True):
        # model_id: 本地路径或 HuggingFace ID "OpenGVLab/InternVL3-2B-Instruct"
        print(f"正在加载OCR模型: {model_id}")

        self.llm = LLM(
            model=model_id,
            max_model_len=8192,
            max_num_seqs=8,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"OCR模型加载完成")

        self.enable_chart_detection = enable_chart_detection
        if self.enable_chart_detection:
            self.chart_detector = MathProblemChartDetector()
            print("图表检测器已启用")

    def ocr_batch(self, image_paths, detect_chart=None):
        """批量OCR识别，分4批推理避免显存溢出"""
        if not image_paths:
            return []

        try:
            from PIL import Image as PILImage

            batch_inputs = []
            query = get_ocr_prompt()

            all_messages = [
                [{"role": "user", "content": f"<image>\n{query}"}]
                for _ in image_paths
            ]

            prompts = self.tokenizer.apply_chat_template(
                all_messages, tokenize=False, add_generation_prompt=True
            )

            for idx, image_path in enumerate(image_paths):
                image = PILImage.open(image_path).convert("RGB").resize(IMAGE_SIZE)
                batch_inputs.append({
                    "prompt": prompts[idx],
                    "multi_modal_data": {"image": image},
                })

            sampling_params = SamplingParams(
                temperature=0.4,
                max_tokens=2048,
                repetition_penalty=1.05,
                frequency_penalty=0.1,
                presence_penalty=0.1,
            )

            # 分4批推理，每批后释放显存
            total = len(batch_inputs)
            batch_size = total // 4
            remainder = total % 4
            all_outputs = []

            for i in range(3):
                start = i * batch_size
                end = start + batch_size
                print(f"OCR第{i+1}批次: {batch_size} 个样本")
                current = batch_inputs[start:end]
                outputs = self.llm.generate(current, sampling_params=sampling_params)
                all_outputs.extend(outputs)
                del current, outputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            start = 3 * batch_size
            last_size = batch_size + remainder
            print(f"OCR第4批次: {last_size} 个样本")
            last = batch_inputs[start:]
            outputs = self.llm.generate(last, sampling_params=sampling_params)
            all_outputs.extend(outputs)
            del last, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            results = []
            should_detect = detect_chart if detect_chart is not None else self.enable_chart_detection

            for output in all_outputs:
                text = output.outputs[0].text.strip()
                result = {'text': text, 'is_chart': None}
                if should_detect and text:
                    result['is_chart'] = self.chart_detector.detect_chart_in_math_problem(text, simple_mode=True)
                results.append(result)

            return results

        except Exception as e:
            print(f"批量OCR识别失败: {str(e)}")
            return [{'text': "", 'is_chart': None} for _ in image_paths]

    def release(self):
        """释放模型资源"""
        for attr in ['chart_detector', 'tokenizer', 'llm']:
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
