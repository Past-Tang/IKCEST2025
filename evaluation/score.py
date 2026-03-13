"""
评分脚本 - 比较预测答案与参考答案

按题型统计正确率，输出详细结果。

用法：
    python score.py <参考答案.jsonl> <预测答案.jsonl>
"""

import json
import sys
from pathlib import Path


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def main(gt_file, pred_file):
    gt_data = load_jsonl(gt_file)
    pred_data = load_jsonl(pred_file)
    print(f"参考答案: {len(gt_data)} 条 | 预测答案: {len(pred_data)} 条")

    gt_dict = {}
    for item in gt_data:
        image = item.get('image', '')
        gt_dict[image] = {
            'answer': item.get('gt_answer', item.get('answer', '')),
            'tag': item.get('tag', ''),
        }

    correct, total = 0, 0
    results = []

    for item in pred_data:
        image = item.get('image', '')
        pred_answer = str(item.get('answer', '')).strip()

        if image not in gt_dict:
            continue

        gt_info = gt_dict[image]
        gt_answer = str(gt_info['answer']).strip()
        is_correct = gt_answer == pred_answer

        if is_correct:
            correct += 1
        total += 1

        results.append({
            'image': image, 'tag': gt_info['tag'],
            'gt_answer': gt_answer, 'pred_answer': pred_answer,
            'correct': is_correct,
        })

    # 详细结果
    print(f"\n{'='*60}")
    for i, r in enumerate(results, 1):
        status = "✓" if r['correct'] else "✗"
        print(f"[{i}/{total}] {status} {r['image']} | {r['tag']} | "
              f"GT={r['gt_answer']} PRED={r['pred_answer']}")

    # 总体统计
    print(f"\n{'='*60}")
    print(f"总题数: {total} | 正确: {correct} | 准确率: {correct/total*100:.2f}%")

    # 按题型统计
    tag_stats = {}
    for r in results:
        tag = r['tag']
        tag_stats.setdefault(tag, {'correct': 0, 'total': 0})
        tag_stats[tag]['total'] += 1
        if r['correct']:
            tag_stats[tag]['correct'] += 1

    print(f"\n按题型:")
    for tag, s in sorted(tag_stats.items()):
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  {tag}: {s['correct']}/{s['total']} = {acc:.2f}%")

    # 保存详细结果
    output_file = Path(pred_file).parent / f"{Path(pred_file).stem}_score.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\n详细结果: {output_file}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python score.py <参考答案.jsonl> <预测答案.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
