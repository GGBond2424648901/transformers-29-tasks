# 📊 文本分类（Text Classification）

## 📖 任务简介

文本分类是自然语言处理中最基础和常见的任务之一，目标是将文本分配到预定义的类别中。这是一个监督学习任务，需要标注的训练数据。

## 🎯 应用场景

- **😊 情感分析**: 判断文本的情感倾向（正面/负面/中性）
- **📧 垃圾邮件过滤**: 识别垃圾邮件和正常邮件
- **📰 新闻分类**: 将新闻分类到不同主题（科技、体育、娱乐等）
- **🏷️ 主题标注**: 为文档自动添加主题标签
- **⚠️ 内容审核**: 识别不当内容、敏感信息

## 📁 文件说明

- `run_classification.py` - 通用文本分类训练脚本
- `run_glue.py` - GLUE 基准测试训练脚本（使用 Trainer）
- `run_glue_no_trainer.py` - GLUE 训练脚本（不使用 Trainer）
- `run_xnli.py` - 跨语言自然语言推理训练
- `requirements.txt` - 依赖包列表
- `README.md` - 本文件

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

文本分类需要标注数据，格式通常为：

```
文本,标签
这部电影真好看,正面
服务态度太差了,负面
价格还可以,中性
```

### 3. 运行训练

#### 使用 Hugging Face 数据集

```bash
python run_classification.py \
    --model_name_or_path bert-base-chinese \
    --dataset_name ydshieh/coco_dataset_script \
    --output_dir ./output \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3
```

#### 使用本地 CSV 文件

```bash
python run_classification.py \
    --model_name_or_path bert-base-chinese \
    --train_file train.csv \
    --validation_file val.csv \
    --text_column_name text \
    --label_column_name label \
    --output_dir ./output \
    --do_train \
    --do_eval
```

## 💡 使用示例

### Pipeline 推理（无需训练）

```python
from transformers import pipeline

# 创建分类器
classifier = pipeline(
    "text-classification",
    model="bert-base-chinese"
)

# 分类单个文本
result = classifier("这部电影真的太精彩了！")
print(result)
# [{'label': 'POSITIVE', 'score': 0.98}]

# 批量分类
texts = [
    "产品质量很好",
    "服务态度太差",
    "价格适中"
]
results = classifier(texts)
```

### 训练自定义分类器

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 1. 加载数据集
dataset = load_dataset("csv", data_files={
    "train": "train.csv",
    "test": "test.csv"
})

# 2. 加载模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  # 类别数量
)

# 3. 数据预处理
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. 训练配置
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 5. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# 6. 开始训练
trainer.train()

# 7. 保存模型
trainer.save_model("./my_classifier")
```

## 🎨 推荐模型

### 中文模型

| 模型 | 大小 | 特点 |
|------|------|------|
| bert-base-chinese | 中 | 通用中文 BERT，效果稳定 |
| hfl/chinese-roberta-wwm-ext | 中 | 中文 RoBERTa，效果更好 |
| hfl/chinese-bert-wwm-ext | 中 | 全词掩码 BERT |
| uer/roberta-base-finetuned-chinanews-chinese | 中 | 针对中文新闻优化 |

### 英文模型

| 模型 | 大小 | 特点 |
|------|------|------|
| bert-base-uncased | 中 | 通用英文 BERT |
| roberta-base | 中 | 英文 RoBERTa，效果更好 |
| distilbert-base-uncased | 小 | 轻量级，速度快 |
| albert-base-v2 | 小 | 参数共享，内存占用小 |

## 📊 数据格式

### CSV 格式

```csv
text,label
这部电影真好看,1
服务态度太差了,0
价格还可以,2
```

### JSON 格式

```json
[
    {"text": "这部电影真好看", "label": 1},
    {"text": "服务态度太差了", "label": 0},
    {"text": "价格还可以", "label": 2}
]
```

### Hugging Face Dataset

```python
from datasets import load_dataset

# 从 Hub 加载
dataset = load_dataset("glue", "sst2")

# 从本地文件加载
dataset = load_dataset("csv", data_files="data.csv")
```

## ⚙️ 重要参数

### 训练参数

```bash
--model_name_or_path bert-base-chinese  # 预训练模型
--num_train_epochs 3                     # 训练轮数
--per_device_train_batch_size 16         # 批次大小
--learning_rate 2e-5                     # 学习率
--weight_decay 0.01                      # 权重衰减
--warmup_steps 500                       # 预热步数
--max_seq_length 128                     # 最大序列长度
```

### 数据参数

```bash
--train_file train.csv                   # 训练文件
--validation_file val.csv                # 验证文件
--test_file test.csv                     # 测试文件
--text_column_name text                  # 文本列名
--label_column_name label                # 标签列名
```

### 输出参数

```bash
--output_dir ./output                    # 输出目录
--save_strategy epoch                    # 保存策略
--save_total_limit 3                     # 保存检查点数量
--load_best_model_at_end                 # 加载最佳模型
```

## 💡 训练技巧

### 1. 数据不平衡

```python
from sklearn.utils.class_weight import compute_class_weight

# 计算类别权重
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)

# 在训练时使用
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_loss=lambda model, inputs: weighted_loss(model, inputs, class_weights)
)
```

### 2. 数据增强

```python
# 使用同义词替换
from nlpaug.augmenter.word import SynonymAug

aug = SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(text)

# 回译增强
# 中文 -> 英文 -> 中文
```

### 3. 超参数调优

```python
from transformers import Trainer

# 使用 Optuna 进行超参数搜索
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 超参数搜索
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    n_trials=10
)
```

## 📈 评估指标

### 常用指标

- **准确率（Accuracy）**: 正确分类的样本比例
- **精确率（Precision）**: 预测为正的样本中真正为正的比例
- **召回率（Recall）**: 真正为正的样本中被预测为正的比例
- **F1 分数**: 精确率和召回率的调和平均

### 计算示例

```python
from sklearn.metrics import classification_report

# 预测
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

# 评估
report = classification_report(
    test_dataset['label'],
    preds,
    target_names=['负面', '中性', '正面']
)
print(report)
```

## ⚠️ 注意事项

1. **数据质量**: 标注数据的质量直接影响模型效果
2. **类别平衡**: 注意各类别样本数量的平衡
3. **文本长度**: 超过模型最大长度的文本会被截断
4. **过拟合**: 小数据集容易过拟合，需要正则化
5. **评估指标**: 根据任务选择合适的评估指标

## 🔗 相关资源

- [GLUE 基准测试](https://gluebenchmark.com/)
- [Hugging Face 文本分类教程](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [情感分析完整示例](../../情感分析/)
