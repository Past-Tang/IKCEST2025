"""
视觉模型推理模块 - InternVL3.5-2B + vLLM

用于处理含图表的数学题。图像统一调整为 720x720。
"""

from vllm import LLM, SamplingParams
from PIL import Image

IMAGE_SIZE = (720, 720)


def generate_answers(model_path, inputs_list, max_tokens=4096, image_size=IMAGE_SIZE):
    """
    批量视觉推理

    Args:
        model_path: 模型路径
        inputs_list: [(image_path, question), ...]
        max_tokens: 最大生成 token 数
        image_size: 图像缩放尺寸
    """
    print("正在加载视觉模型...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        gpu_memory_utilization=0.6,
        max_num_seqs=4,
    )

    tokenizer = llm.get_tokenizer()

    all_messages = [
        [{"role": "user", "content": f"<image>\n{question}"}]
        for _, question in inputs_list
    ]

    prompts = tokenizer.apply_chat_template(
        all_messages, tokenize=False, add_generation_prompt=True
    )

    batch_inputs = []
    target_size = image_size or IMAGE_SIZE
    for idx, (image_path, _) in enumerate(inputs_list):
        image = Image.open(image_path).convert("RGB")
        if target_size:
            image = image.resize(target_size)
        batch_inputs.append({
            "prompt": prompts[idx],
            "multi_modal_data": {"image": image},
        })

    sampling_params = SamplingParams(temperature=0.6, max_tokens=max_tokens)
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    return [output.outputs[0].text for output in outputs]
