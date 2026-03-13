"""
过滤验证一致的样本

保留 verification_result == CONSISTENT 的样本，
删除 INCONSISTENT / UNCERTAIN / SKIPPED。

用法：
    python filter_consistent.py <verified.jsonl> <filtered.jsonl>
"""

import json
import sys


def filter_consistent(input_jsonl, output_jsonl):
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    filtered = [item for item in data if item.get('verification_result') == 'CONSISTENT']

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"保留 {len(filtered)}/{len(data)} 条 ({len(filtered)/len(data)*100:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python filter_consistent.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    filter_consistent(sys.argv[1], sys.argv[2])
