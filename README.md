# DoTMI: Document-Level Token-Only Membership Inference for Large Language Models

## 📋 项目概述

DoTMI（Document-level Token-only Membership Inference）是一种在黑盒环境下检测文档是否被用于大语言模型训练的方法。该方法仅通过访问模型生成的token，就能有效地进行文档级别的成员推断。

### 核心创新点
- **语义感知的文档分割**：保持每个片段的语义完整性
- **关键段落筛选**：两种高效的方法选择信息密度高的段落
- **Cloze-style任务**：通过填空任务评估模型的记忆程度
- **双特征融合**：结合SimHash语义相似度和近似困惑度

## 📁 文件结构说明

### 核心实验代码

| 文件名 | 功能描述 |
|--------|----------|
| `main.py` | **主运行脚本**。支持运行DoTMI和所有baseline方法，支持Project Gutenberg和ArXiv数据集 |
| `mask.py` | **核心功能模块**。实现mask/fill、SimHash相似度计算、PPL计算、所有baseline方法 |
| `split.py` | **文档分割模块**。基于句子边界将长文档分割成语义完整的片段 |
| `key_segment_filter.py` | **关键段落筛选模块**。实现两种筛选方法：①基于语句独特性 ②基于专有名词密度 |
| `membership_detection.py` | **评估模块**。计算AUC、TPR@FPR、最优阈值搜索 |
| `process_mixed_dataset.py` | **效率对比脚本**。用于评估不同筛选方法的效率 |

### 数据和配置

| 文件/目录 | 说明 |
|-----------|------|
| `data/` | 数据目录（需要用户自行准备数据） |

## 🚀 快速开始

### 1. 环境依赖

```bash
pip install torch transformers numpy scikit-learn matplotlib nltk simhash
```

### 2. 下载NLTK数据

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

### 3. 准备数据

#### Project Gutenberg数据集
```
data/
├── raw_gutenberg/
│   ├── pg19_downloads/      # 成员数据（训练数据，label=1）
│   │   ├── book1.txt
│   │   ├── book2.txt
│   │   └── ...
│   └── data/                # 非成员数据（label=0）
│       ├── book1.txt
│       ├── book2.txt
│       └── ...
```

#### ArXiv数据集
```
data/
└── raw_arxiv/
    ├── member/              # 成员数据（2023年3月之前发表的论文，label=1）
    │   ├── paper1.txt
    │   ├── paper2.txt
    │   └── ...
    └── non_member/          # 非成员数据（2023年3月之后发表的论文，label=0）
        ├── paper1.txt
        ├── paper2.txt
        └── ...
```

**数据要求**：
- 每个文档至少5000个token
- 文本格式为UTF-8编码的纯文本文件（.txt）
- ArXiv论文需要从PDF/TeX转换为纯文本

### 4. 配置模型路径

修改 `mask.py` 中的模型路径：

```python
# 第9-10行
tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL_PATH", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("YOUR_MODEL_PATH")
```

**支持的模型**（与论文一致）：
- LLaMA2 (7B, 13B, 70B)
- Falcon-7B-Instruct
- OPT-6.7B
- Qwen3-8B

## 💻 运行实验

### 基础用法

```bash
# 运行所有方法（DoTMI + baselines）在Project Gutenberg上
python main.py --run_gutenberg --modes dotmi_uniqueness dotmi_density ppl min_k zlib lowercase

# 运行所有方法在ArXiv上
python main.py --run_arxiv --modes dotmi_uniqueness dotmi_density ppl min_k zlib lowercase

# 同时运行两个数据集
python main.py --run_gutenberg --run_arxiv --modes dotmi_uniqueness dotmi_density ppl min_k
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | 42 | 随机种子 |
| `--max_files` | 100 | 每个类别最多处理的文档数 |
| `--min_tokens` | 5000 | 文档最小token数 |
| `--max_tokens` | 512 | 每个片段的最大token数（上下文长度） |
| `--top_ratio` | 0.2 | 关键段落选择比例（前20%） |
| `--mask_ratio` | 0.15 | Mask token的比例（15%） |
| `--modes` | 多个 | 运行的模式（见下方） |
| `--run_gutenberg` | False | 是否运行Gutenberg实验 |
| `--run_arxiv` | False | 是否运行ArXiv实验 |
| `--g_member_dir` | - | Gutenberg成员数据目录 |
| `--g_nonmember_dir` | - | Gutenberg非成员数据目录 |
| `--a_member_dir` | - | ArXiv成员数据目录 |
| `--a_nonmember_dir` | - | ArXiv非成员数据目录 |

### 运行模式说明

| 模式 | 描述 | 输出特征维度 |
|------|------|-------------|
| `dotmi_no_filter` | DoTMI without key segment selection | 2 (SimHash + PPL) |
| `dotmi_uniqueness` | DoTMI + Method 1 (uniqueness-based) | 2 (SimHash + PPL) |
| `dotmi_density` | DoTMI + Method 2 (density-based) | 2 (SimHash + PPL) |
| `ppl` | Perplexity baseline | 1 |
| `min_k` | Min-K% Prob baseline | 1 |
| `zlib` | zlib compression baseline | 1 |
| `lowercase` | lowercase baseline | 1 |
| `prefix_suffix_ppl` | Prefix→Suffix baseline | 1 |

### 输出文件

运行后会生成以下文件：

```
{dataset}_features_{mode}.txt    # 特征向量
{dataset}_labels_{mode}.txt      # 标签（1=成员，0=非成员）
```

### 评估结果

程序会自动计算并显示：
- 最优阈值 (θ_d, θ_ppl)
- AUC分数

### 效率对比实验

使用 `process_mixed_dataset.py` 进行效率对比：

```bash
python process_mixed_dataset.py
```

该脚本会输出：
- 筛选耗时
- 段落减少比例
- 平均处理时间

## 📊 论文实验复现指南

### 1. 主实验

DoTMI与baselines在不同模型上的AUC对比。

**复现步骤**：
```bash
# 对每个模型修改mask.py中的路径，然后运行：
python main.py --run_gutenberg --run_arxiv \
    --modes dotmi_uniqueness dotmi_density ppl min_k zlib lowercase \
    --max_files 1000
```

### 2. 基线对比

对比DoTMI与滑动窗口方法的对比。

**实现方式**：调整 `--max_tokens` 参数并对比性能

### 3. 模型规模影响

在不同规模的LLaMA2模型上运行：

```bash
# LLaMA2-7B
python main.py --run_gutenberg --modes dotmi_uniqueness

# LLaMA2-13B（修改模型路径后）
python main.py --run_gutenberg --modes dotmi_uniqueness

# LLaMA2-70B（修改模型路径后）
python main.py --run_gutenberg --modes dotmi_uniqueness
```

### 4. 上下文长度影响

```bash
for len in 256 512 1024 2048; do
    python main.py --run_gutenberg --run_arxiv \
        --modes dotmi_uniqueness ppl min_k \
        --max_tokens $len
done
```

### 5. Mask Ratio影响

```bash
for ratio in 0.05 0.10 0.15 0.25 0.30 0.45; do
    python main.py --run_gutenberg --run_arxiv \
        --modes dotmi_uniqueness \
        --mask_ratio $ratio
done
```

### 6. 效率对比

```bash
python process_mixed_dataset.py
```

### 7. 消融实验

需要自定义实现，基本思路：
- 对比不同特征组合（仅SimHash vs 仅PPL vs 两者）
- 对比不同选择策略（全部 vs 随机20% vs Method 1 vs Method 2）

## ⚠️ 注意事项

### 1. 硬编码路径
以下路径需要根据你的环境修改：
- 模型路径：`mask.py` 第9-10行
- 数据路径：`main.py` 默认参数或命令行参数
- process_mixed_dataset.py 中的数据路径

### 2. 数据集准备
- Project Gutenberg数据需要自行下载并整理
- ArXiv论文需要转换为txt格式（推荐使用 `pdftotext` 或类似工具）

### 3. 计算资源
- 推荐使用GPU加速（至少16GB显存）
- 处理1000本书需要数小时（取决于模型大小）

### 4. 代码与论文的差异

**已实现但与论文略有差异的部分**：
- SimHash实现简化（未使用TF-IDF加权，但核心功能一致）
- 部分高级实验需要额外实现（如相似度指标对比实验）

**未完全实现的实验**：
- Cosine相似度和Rouge-N对比实验（Table 3）
- 自动化生成所有图表的脚本

## 🔧 扩展和定制

### 添加新的相似度指标

在 `mask.py` 中添加新函数：

```python
def cosine_similarity(text1, text2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
```

### 添加新的模型

只需修改 `mask.py` 中的模型路径：

```python
# 替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model")
model = AutoModelForCausalLM.from_pretrained("/path/to/your/model")
```

### 自定义数据集

遵循以下格式：
```
your_dataset/
├── member/
│   ├── doc1.txt
│   └── ...
└── non_member/
    ├── doc1.txt
    └── ...
```

然后运行：
```bash
python main.py --run_gutenberg \
    --g_member_dir your_dataset/member \
    --g_nonmember_dir your_dataset/non_member
```

## 🐛 常见问题

**Q1: 运行时提示找不到模型**
A: 检查 `mask.py` 中的模型路径是否正确

**Q2: 数据集为空**
A: 确保数据目录中有 `.txt` 文件，且文件编码为UTF-8

**Q3: 内存不足**
A: 减小 `--max_files` 或 `--max_tokens` 参数

**Q4: NLTK数据下载失败**
A: 手动下载或设置代理

**Q5: 结果与论文不一致**
A: 检查数据集、模型、参数设置是否完全一致


## 附录：代码与论文对照检查

### ✅ 已完整实现
- [x] 文档级别成员推断框架
- [x] 语义感知的文档分割
- [x] 两种关键段落筛选方法
- [x] Cloze-style任务设计
- [x] SimHash相似度计算
- [x] 近似困惑度计算
- [x] 双特征融合和阈值优化
- [x] 所有baseline方法（PPL, Min-K, zlib, lowercase, prefix-suffix）
- [x] Project Gutenberg和ArXiv数据集支持
- [x] AUC评估和TPR@FPR计算
- [x] 可配置的实验参数

### ⚠️ 部分实现（核心功能完整）
- [~] SimHash的完整TF-IDF加权（简化版本，功能一致）
- [~] 效率对比实验（基础实现，需整合）

### ❌ 需要用户额外实现
- [ ] Cosine相似度和Rouge-N对比实验（Table 3）
- [ ] 自动化图表生成脚本
- [ ] 多模型批量运行脚本

### 数据集覆盖情况
- ✅ **Project Gutenberg**: 完整支持
- ✅ **ArXiv**: 代码支持，需用户准备txt格式数据

### 模型支持情况
- ✅ **LLaMA2**: 支持（需修改路径）
- ✅ **Falcon**: 支持（需修改路径）
- ✅ **OPT**: 支持（需修改路径）
- ✅ **Qwen3**: 支持（需修改路径）

总体而言，代码完整实现了论文的核心方法和主要实验，能够复现论文的主要结果。
