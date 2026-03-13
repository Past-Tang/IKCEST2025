import os
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

def generate_answers(model_id, prompts_list, max_tokens=11520):
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

    # ==================== 修改部分：分成4批进行推理 ====================
    total_samples = len(texts)
    batch_size = total_samples // 4
    
    # 计算每个批次的大小
    batch_sizes = []
    remainder = total_samples % 4
    
    # 分配批次大小，确保所有样本都被处理
    for i in range(4):
        if i < remainder:
            batch_sizes.append(batch_size + 1)
        else:
            batch_sizes.append(batch_size)
    
    all_outputs = []
    start_idx = 0
    
    # 处理4个批次
    for i in range(4):
        current_batch_size = batch_sizes[i]
        end_idx = start_idx + current_batch_size
        
        print(f"开始第{i+1}批次推理: {current_batch_size} 个样本")
        current_batch = texts[start_idx:end_idx]
        
        if current_batch:  # 确保批次不为空
            current_outputs = llm.generate(current_batch, sampling_params)
            all_outputs.extend(current_outputs)
        
        start_idx = end_idx
    
    outputs = all_outputs
    # ==================== 修改部分结束 ====================
    
    answers = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        answers.append(generated_text)
    
    return answers

if __name__ == "__main__":
    model_id = "./OpenMath-Nemotron-1.5B"
    
    # Example batch prompts
    prompt1 = [
        {
            "role": "user", 
            "content": "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}." + 
            r'''\[\lim_{x\to0}\frac{e^{x}-e^{x\cos x}}{x\ln(1+x^{2})}\]'''},
    ]
    
    prompt2 = [
        {
            "role": "user", 
            "content": "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}." + 
            r'''\[\lim_{x\to\infty}\frac{\ln x}{x^{1/2}}\]'''},
    ]
    
    prompt3 = [
        {
            "role": "user", 
            "content": "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}." + 
            r'''\[\int_0^1 x^2 \, dx\]'''},
    ]
    
    prompts_list = [prompt1, prompt2, prompt3]
    
    answers = generate_answers(model_id, prompts_list)
    
    for i, answer in enumerate(answers):
        print(f"Answer {i+1}: {answer}\n")