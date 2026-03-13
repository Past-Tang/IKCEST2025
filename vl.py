#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VL 模块 - 使用 Ovis2-2B + vLLM 进行批量视觉推理
"""

import os
from vllm import LLM, SamplingParams
from PIL import Image
from prompt import get_vl_prompt, MATH_SOLVE_BASE

# ==================== 全局配置 ====================
# 图像尺寸：所有输入图像将被调整为此尺寸
IMAGE_SIZE = (720, 720)

def generate_answers(model_path: str, inputs_list: list, max_tokens=4096, image_size: tuple = IMAGE_SIZE):
    """
    批量视觉推理函数
    
    Args:
        model_path: 模型路径
        inputs_list: 批量输入列表，每个元素为 (image_path: str, question: str) 元组
        max_tokens: 最大生成 token 数
        image_size: 可选，图像输入的尺寸 (width, height)，如果提供则调整图像大小，默认为 None
    
    Returns:
        list: 模型的回答列表
    """
    # 加载模型（针对 V100-32GB 优化配置）
    print("正在加载视觉模型（Ovis2-2B + vLLM）...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1, "video": 0},  # 禁用视频，只用图像
        gpu_memory_utilization=0.6,  # 降低到 0.6 避免 OOM
        max_num_seqs=4,  # 限制最大并发序列数
        
        
         
    )
    
    # 获取 tokenizer
    tokenizer = llm.get_tokenizer()
    
    # 准备批量输入（参考 vLLM 官方 Ovis 示例）
    batch_inputs = []
    
    # 构建所有消息（需要包含 <image> 标记）
    all_messages = [
        [{"role": "user", "content": f"<image>\n{question}"}] 
        for _, question in inputs_list
    ]
    
    # 批量生成提示词
    prompts = tokenizer.apply_chat_template(
        all_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 构建批量输入
    for idx, (image_path, question) in enumerate(inputs_list):
        prompt = prompts[idx]
        
        # 加载图像为 PIL Image 对象
        image = Image.open(image_path).convert("RGB")
        
        # 调整图像尺寸为 720x720（如果未指定，使用默认 IMAGE_SIZE）
        target_size = image_size if image_size is not None else IMAGE_SIZE
        if target_size:
            image = image.resize(target_size)
        
        # 使用 PIL Image 对象（不是路径！）
        batch_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        })
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=max_tokens
    )
    
    # 推理
    outputs = llm.generate(
        batch_inputs,
        sampling_params=sampling_params,
    )
    
    # 提取回答
    answers = []
    for output in outputs:
        answer = output.outputs[0].text
        answers.append(answer)
    
    return answers


def main():
    # 模型路径
    model_path = "./Ovis2-2B"
    
    # 从 prompt.py 获取统一提示词
    math_prompt = MATH_SOLVE_BASE
    
    # 示例批量输入
    input1 = (
        "/home/aistudio/work/my_test/images/1_3B0F499C67ACF75E6B64E74641335470.JPEG",
        math_prompt
    )
    
    input2 = (
        "/home/aistudio/work/my_test/images/1_3B0F499C67ACF75E6B64E74641335470.JPEG",
        math_prompt
    )
    
    input3 = (
        "/home/aistudio/work/my_test/images/1_3B0F499C67ACF75E6B64E74641335470.JPEG",
        math_prompt
    )
    
    inputs_list = [input1, input2, input3]
    
    print("批量输入:")
    for i, (img_path, q) in enumerate(inputs_list):
        print(f"Input {i+1}: Image={img_path}, Question={q}")
    print("\n" + "=" * 80)
    
    # 使用全局 IMAGE_SIZE 或传递自定义尺寸
    answers = generate_answers(model_path, inputs_list)
    
    for i, answer in enumerate(answers):
        print(f"Answer {i+1}: {answer}\n")
    print("=" * 80)


if __name__ == "__main__":
    main()