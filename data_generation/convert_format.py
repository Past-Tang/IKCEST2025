"""
数据格式转换 - 将推理输出转为 SFT messages 格式

输入格式: {"image": "xxx.jpg", "modelprint": "..."}
输出格式: {"messages": [{"role": "user", "content": "<image>"}, {"role": "assistant", "content": "..."}], "images": ["xxx.jpg"]}

用法：
    python convert_format.py <input.jsonl> <output.jsonl>
"""

import json
import sys


def convert_format(input_file, output_file):
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                new_data = {
                    "messages": [
                        {"role": "user", "content": "<image>"},
                        {"role": "assistant", "content": data.get("modelprint", "")}
                    ],
                    "images": [data.get("image", "")]
                }
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                count += 1
            except json.JSONDecodeError:
                continue

    print(f"转换完成: {count} 条")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python convert_format.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    convert_format(sys.argv[1], sys.argv[2])
