"""
SFT → GRPO 数据集格式转换

从 SFT 数据（含 assistant 回复）中：
    1. 提取 \\boxed{} 作为 solution
    2. 移除 assistant 回复（GRPO 自动生成）
    3. 保留 images 等多模态字段

用法：
    python preprocess_dataset.py input.jsonl output.jsonl
    python preprocess_dataset.py input.jsonl output.jsonl --sample 6000
"""

import json
import re
import sys
import random
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm


class DatasetPreprocessor:

    def __init__(self, keep_assistant: bool = False):
        self.keep_assistant = keep_assistant
        self.stats = {'total': 0, 'success': 0, 'no_assistant': 0, 'no_boxed': 0, 'invalid_format': 0}

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        return matches[-1].strip() if matches else None

    def process_single_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.stats['total'] += 1

        if 'messages' not in sample:
            self.stats['invalid_format'] += 1
            return None

        assistant_content = None
        user_messages = []
        for msg in sample['messages']:
            if msg.get('role') == 'assistant':
                assistant_content = msg.get('content', '')
            elif msg.get('role') in ['user', 'system']:
                user_messages.append(msg)

        if assistant_content is None:
            self.stats['no_assistant'] += 1
            return None

        answer = self.extract_boxed_answer(assistant_content)
        if answer is None:
            self.stats['no_boxed'] += 1
            return None

        new_sample = {
            'messages': sample['messages'].copy() if self.keep_assistant else user_messages,
            'solution': f'\\boxed{{{answer}}}'
        }
        for key, value in sample.items():
            if key not in ['messages', 'solution']:
                new_sample[key] = value

        self.stats['success'] += 1
        return new_sample

    def process_dataset(self, input_path: str, output_path: str,
                        sample_size: Optional[int] = None, random_seed: int = 42):
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f if line.strip()]

        if sample_size and sample_size < len(all_lines):
            random.seed(random_seed)
            all_lines = random.sample(all_lines, sample_size)

        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in tqdm(all_lines, desc="处理数据"):
                try:
                    sample = json.loads(line)
                    processed = self.process_single_sample(sample)
                    if processed:
                        fout.write(json.dumps(processed, ensure_ascii=False) + '\n')
                except (json.JSONDecodeError, Exception) as e:
                    self.stats['invalid_format'] += 1

        print(f"\n处理统计: 总计={self.stats['total']} 成功={self.stats['success']} "
              f"无assistant={self.stats['no_assistant']} 无boxed={self.stats['no_boxed']}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python preprocess_dataset.py <input.jsonl> <output.jsonl> [--sample N] [--seed N]")
        sys.exit(1)

    sample_size = None
    random_seed = 42
    if '--sample' in sys.argv:
        idx = sys.argv.index('--sample')
        sample_size = int(sys.argv[idx + 1])
    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        random_seed = int(sys.argv[idx + 1])

    preprocessor = DatasetPreprocessor(keep_assistant='--keep-assistant' in sys.argv)
    preprocessor.process_dataset(sys.argv[1], sys.argv[2], sample_size=sample_size, random_seed=random_seed)
