# 2025西交大数据竞赛 · 数学题图像理解方案

**得分：67.4 分**（排行榜前列）

## 比赛简介

本方案针对 **2025西安交通大学数据竞赛** 中的数学题图像理解任务。  
任务要求：给定包含数学题的图像，自动识别题目内容并输出正确答案。

**题型：**
- 选择题：输出选项字母（A / B / C / D）
- 填空题：输出保留六位小数的浮点数
- 计算应用题：输出保留六位小数的浮点数

**硬件限制：** 单张 V100-32GB GPU

---

## 方案概述：分而治之三阶段流水线

核心思想是**先 OCR 识别题目文本，再根据是否含图表分流到不同模型推理**，避免用同一模型处理所有题目造成的能力浪费。

```
输入图像
    │
    ▼
【阶段1】OCR识别（InternVL3.5-2B）
    ├─ 提取题目文本
    └─ 检测是否含图表
         │
    ┌────┴────┐
    │         │
 无图表      有图表
    │         │
    ▼         ▼
【阶段2】     【阶段3】
文本推理      视觉推理
OpenMath-    InternVL3.5-2B
Nemotron-1.5B
    │         │
    └────┬────┘
         ▼
      合并输出
```

### 分流逻辑

- **有图表**（含坐标系、函数图像、几何图形等）→ 使用多模态视觉模型直接看图解题
- **无图表**（纯文字计算/填空）→ 使用专门的数学文本推理模型，效率更高

---

## 模型选择

| 用途 | 模型 | 说明 |
|------|------|------|
| OCR + 视觉推理 | [InternVL3.5-2B-Instruct](https://huggingface.co/OpenGVLab/InternVL2_5-2B) | 轻量级多模态模型，适配 V100 显存 |
| 文本数学推理 | [OpenMath-Nemotron-1.5B](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B) | NVIDIA 专为数学推理训练的小模型 |

两个模型总参数约 3.5B，串行加载/释放，显存峰值不超过 V100-32GB 限制。

---

## 文件结构

```
.
├── run.py              # 主推理入口，三阶段流程控制
├── ocr.py              # 阶段1：OCR识别 + 图表检测
├── text.py             # 阶段2：文本模型批量推理
├── vl.py               # 阶段3：视觉模型批量推理
├── chart_detector.py   # 图表检测器（基于关键词匹配）
├── prompt.py           # 提示词管理（按题型分类）
├── answer.py           # 答案提取（math-verify + 正则兜底）
├── run.sh              # 一键运行脚本（竞赛环境）
├── build_env.sh        # 环境安装脚本
└── requirements.txt    # 依赖列表
```

---

## 快速开始

### 1. 安装环境

```bash
bash build_env.sh
```

或手动安装：

```bash
pip install vllm==0.10.2 transformers==4.56.2 math_verify sympy latex2sympy2_extended==1.10.2 qwen_vl_utils
```

### 2. 准备模型

将以下模型下载到对应目录：

```
./InternVL3_5-2B-Instruct/   # OCR + 视觉推理模型
./OpenMath-Nemotron-1.5B/    # 文本数学推理模型
```

### 3. 运行推理

**完整三阶段（推荐）：**

```bash
export IMAGE_INPUT_DIR=/path/to/images
export QUERY_PATH=/path/to/input.jsonl
export OUPUT_PATH=./output.jsonl

bash run.sh
```

**分阶段运行：**

```bash
# 阶段1：OCR识别
python run.py stage1 <image_dir> <input.jsonl> <ocr_result.jsonl>

# 阶段2：文本模型推理（无图表题）
python run.py stage2 <image_dir> <ocr_result.jsonl> <text_result.jsonl>

# 阶段3：视觉模型推理（有图表题）+ 合并
python run.py stage3 <image_dir> <ocr_result.jsonl> <text_result.jsonl> <output.jsonl>
```

### 输入格式（JSONL）

每行一个 JSON 对象：

```json
{"id": 1, "image": "001.jpg", "tag": "选择题"}
{"id": 2, "image": "002.jpg", "tag": "填空题"}
{"id": 3, "image": "003.jpg", "tag": "计算应用题"}
```

### 输出格式（JSONL）

```json
{"id": 1, "image": "001.jpg", "tag": "选择题", "answer": "B"}
{"id": 2, "image": "002.jpg", "tag": "填空题", "answer": "3.141593"}
```

---

## 工程细节

### V100 显存优化

V100-32GB 运行 vLLM 需要一些特殊配置（写在 `run.sh` 环境变量中）：

```bash
export VLLM_USE_V1=0                    # 使用 V0 引擎（V1 在 V100 不稳定）
export VLLM_ATTENTION_BACKEND=XFORMERS  # 用 XFormers 替代 Flash Attention
export VLLM_USE_TRITON_FLASH_ATTN=0     # 禁用 Triton Flash Attention
export VLLM_TORCH_COMPILE_LEVEL=0       # 禁用 torch.compile
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

三个阶段**串行执行，每阶段结束后彻底释放显存**，避免 OOM。

### 图表检测

`chart_detector.py` 基于 OCR 文本做关键词匹配，检测是否含图表：

- 图表关键词：图、坐标、函数图像、如图所示……（权重 0.3）
- 图形引用：如图X所示、由图可知……（权重 0.3）
- 几何测量词：边长、面积、体积……（权重 0.2）
- 置信度阈值：≥ 0.3 判定为有图表

### 答案提取

`answer.py` 使用 `math-verify` 库解析 LaTeX 数学表达式，多重降级策略：

1. `\boxed{}` 正则提取
2. `math_verify.parse()` 解析 SymPy 对象 → `evalf()` 数值化
3. 分数字符串处理（如 `1/3`）
4. 兜底默认答案（选择题→C，数值题→1.000000）

---

## 历史版本得分对比

| 版本 | 方案 | 得分 |
|------|------|------|
| baseline | Qwen2.5-VL-3B 单模型端到端 | 60.5 |
| baseline2 | Qwen2.5-VL-3B + 提示词优化 | 65.1 |
| baseline3 | Qwen2.5-VL-3B + 答案提取优化 | 64.6 |
| **本方案** | **InternVL3.5-2B（OCR）+ OpenMath-Nemotron（文本）三阶段** | **67.4** |

---

## 进阶思路（未实现）

方案文档中记录了若干未来改进方向：

- **数据合成**：将 [LIMO](https://github.com/GAIR-NLP/LIMO)、[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) 中的 TeX 公式渲染为图像，扩充训练数据
- **SFT → RL 训练流程**：先 SFT 学习数学能力和规范推理格式，再用 RL 增强泛化能力
- **思维链数据合成**：基于 GLM4.5-V 对真实题目合成 `<think>` 推理过程

---

## License

MIT
