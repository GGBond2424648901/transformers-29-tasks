# 📄 文本摘要（Text Summarization）

## 📖 任务简介

文本摘要是将长文本压缩成简短摘要的任务，保留原文的关键信息。分为抽取式摘要（从原文中选择句子）和生成式摘要（生成新的句子）。

## 🎯 应用场景

- **📰 新闻摘要**: 自动生成新闻标题和摘要
- **📚 文档总结**: 长文档的快速概览
- **📧 邮件摘要**: 提取邮件要点
- **🎬 视频字幕**: 生成视频内容摘要
- **📊 报告生成**: 自动生成会议纪要、研究报告摘要

## 📁 文件说明

- `run_summarization.py` - 使用 Trainer API 训练摘要模型
- `run_summarization_no_trainer.py` - 不使用 Trainer 训练
- `requirements.txt` - 依赖包列表
- `README.md` - 本文件

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行训练

#### 使用 CNN/DailyMail 数据集

```bash
python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --output_dir ./output \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --predict_with_generate
```

#### 使用本地数据

```bash
python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --train_file train.json \
    --validation_file val.json \
    --text_column article \
    --summary_column summary \
    --output_dir ./output \
    --do_train \
    --do_eval
```

## 💡 使用示例

### Pipeline 推理

```python
from transformers import pipeline

# 创建摘要生成器
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# 生成摘要
article = """
人工智能（AI）正在改变我们的生活方式。从智能手机到自动驾驶汽车，
AI 技术无处不在。机器学习算法可以分析大量数据，识别模式，
并做出预测。深度学习是机器学习的一个分支，使用神经网络来模拟
人脑的工作方式。随着计算能力的提升和数据的增加，AI 的应用
将会更加广泛。
"""

summary = summarizer(
    article,
    max_length=50,
    min_length=10,
    do_sample=False
)

print(summary[0]['summary_text'])
# 输出：人工智能正在改变生活方式，应用广泛。
```

### 批量摘要

```python
articles = [
    "长文本1...",
    "长文本2...",
    "长文本3..."
]

summaries = summarizer(
    articles,
    max_length=50,
    min_length=10,
    batch_size=8
)

for i, summary in enumerate(summaries):
    print(f"文章 {i+1} 摘要: {summary['summary_text']}")
```

### 训练自定义摘要模型

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# 1. 加载数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 2. 加载模型和分词器
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. 数据预处理
def preprocess_function(examples):
    inputs = examples["article"]
    targets = examples["highlights"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=1024,
        truncation=True
    )
    
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True
)

# 4. 训练配置
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    predict_with_generate=True,
)

# 5. 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# 6. 创建 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# 7. 开始训练
trainer.train()
```

## 🎨 推荐模型

### 英文模型

| 模型 | 大小 | 特点 |
|------|------|------|
| facebook/bart-large-cnn | 大 | CNN/DailyMail 训练，新闻摘要效果好 |
| google/pegasus-cnn_dailymail | 大 | 专门为摘要设计 |
| t5-base | 中 | 通用 Seq2Seq 模型 |
| google/pegasus-xsum | 大 | 极简摘要风格 |

### 中文模型

| 模型 | 特点 |
|------|------|
| csebuetnlp/mT5_multilingual_XLSum | 多语言摘要，支持中文 |
| fnlp/bart-base-chinese | 中文 BART |

## 📊 数据格式

### JSON 格式

```json
[
    {
        "article": "长文本内容...",
        "summary": "摘要内容"
    }
]
```

### CSV 格式

```csv
article,summary
"长文本内容...","摘要内容"
```

## ⚙️ 重要参数

### 生成参数

```python
summarizer(
    text,
    max_length=130,        # 最大摘要长度
    min_length=30,         # 最小摘要长度
    do_sample=False,       # 是否采样
    num_beams=4,           # Beam search 数量
    length_penalty=2.0,    # 长度惩罚
    early_stopping=True    # 早停
)
```

### 训练参数

```bash
--max_source_length 1024       # 输入最大长度
--max_target_length 128        # 摘要最大长度
--val_max_target_length 128    # 验证时摘要最大长度
--num_beams 4                  # Beam search
--predict_with_generate        # 使用生成模式评估
```

## 💡 训练技巧

### 1. 控制摘要长度

```python
# 短摘要
summary = summarizer(text, max_length=50, min_length=10)

# 长摘要
summary = summarizer(text, max_length=200, min_length=50)
```

### 2. 提高摘要质量

```python
# 使用 Beam Search
summary = summarizer(
    text,
    num_beams=5,           # 增加 beam 数量
    length_penalty=2.0,    # 长度惩罚
    early_stopping=True
)

# 使用采样
summary = summarizer(
    text,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)
```

### 3. 处理长文本

```python
# 分段摘要
def summarize_long_text(text, max_chunk_length=1000):
    # 分割文本
    chunks = [text[i:i+max_chunk_length] 
              for i in range(0, len(text), max_chunk_length)]
    
    # 分别摘要
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=100)
        summaries.append(summary[0]['summary_text'])
    
    # 合并摘要
    combined = " ".join(summaries)
    
    # 再次摘要
    final_summary = summarizer(combined, max_length=150)
    return final_summary[0]['summary_text']
```

## 📈 评估指标

### ROUGE 分数

```python
from datasets import load_metric

rouge = load_metric("rouge")

predictions = ["预测的摘要1", "预测的摘要2"]
references = ["参考摘要1", "参考摘要2"]

results = rouge.compute(
    predictions=predictions,
    references=references
)

print(f"ROUGE-1: {results['rouge1'].mid.fmeasure:.4f}")
print(f"ROUGE-2: {results['rouge2'].mid.fmeasure:.4f}")
print(f"ROUGE-L: {results['rougeL'].mid.fmeasure:.4f}")
```

### 指标说明

- **ROUGE-1**: 单词重叠
- **ROUGE-2**: 双词组重叠
- **ROUGE-L**: 最长公共子序列

## 🎯 应用示例

### 1. 新闻摘要

```python
news = """
【科技新闻】今天，某科技公司发布了最新的人工智能模型...
（长篇新闻内容）
"""

summary = summarizer(news, max_length=100)
print(f"新闻摘要: {summary[0]['summary_text']}")
```

### 2. 会议纪要

```python
meeting_notes = """
会议时间：2026年1月31日
参会人员：...
会议内容：...
（详细会议记录）
"""

summary = summarizer(meeting_notes, max_length=150)
print(f"会议要点: {summary[0]['summary_text']}")
```

### 3. 文档总结

```python
document = """
研究报告：人工智能在医疗领域的应用
摘要：...
引言：...
方法：...
结果：...
讨论：...
结论：...
"""

summary = summarizer(document, max_length=200)
print(f"文档摘要: {summary[0]['summary_text']}")
```

## ⚠️ 注意事项

1. **计算资源**: 摘要模型通常较大，需要 GPU 加速
2. **文本长度**: 注意输入文本的最大长度限制
3. **摘要质量**: 生成的摘要可能不完美，需要人工审核
4. **语言支持**: 大多数模型针对英文优化，中文效果可能较差
5. **事实准确性**: 生成式摘要可能产生不准确的信息

## 🔗 相关资源

- [BART 论文](https://arxiv.org/abs/1910.13461)
- [PEGASUS 论文](https://arxiv.org/abs/1912.08777)
- [CNN/DailyMail 数据集](https://huggingface.co/datasets/cnn_dailymail)
- [ROUGE 评估指标](https://en.wikipedia.org/wiki/ROUGE_(metric))
