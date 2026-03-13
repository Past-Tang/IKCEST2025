"""
文本模型推理模块 - OpenMath-Nemotron-1.5B

用于处理纯文字数学题（OCR 检测为无图表的题目）。
分4批推理以避免 V100-32GB 显存溢出。
"""

from vllm import LLM, SamplingParams
from transformers import AutoProcessor


def generate_answers(model_id, prompts_list, max_tokens=11520):
    """批量文本推理"""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    texts = []
    for messages in prompts_list:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        max_num_seqs=4,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    # 分4批推理
    total = len(texts)
    batch_size = total // 4
    remainder = total % 4

    batch_sizes = []
    for i in range(4):
        batch_sizes.append(batch_size + (1 if i < remainder else 0))

    all_outputs = []
    start = 0

    for i in range(4):
        end = start + batch_sizes[i]
        print(f"文本模型第{i+1}批次: {batch_sizes[i]} 个样本")
        batch = texts[start:end]
        if batch:
            outputs = llm.generate(batch, sampling_params)
            all_outputs.extend(outputs)
        start = end

    return [output.outputs[0].text.strip() for output in all_outputs]
