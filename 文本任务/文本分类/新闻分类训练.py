#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻分类器训练示例
使用 BERT 训练一个6分类的新闻分类器
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch

# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'output', 'news_classifier')

# 类别标签
LABELS = {
    0: "科技",
    1: "体育", 
    2: "娱乐",
    3: "财经",
    4: "社会",
    5: "政治"
}

print("=" * 70)
print("📰 新闻分类器训练")
print("=" * 70)

# 1. 加载数据
print("\n📊 步骤 1/6: 加载数据...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# 删除包含NaN的行
train_df = train_df.dropna()
test_df = test_df.dropna()

# 确保标签是整数类型
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

print(f"   训练集: {len(train_df)} 条")
print(f"   测试集: {len(test_df)} 条")
print(f"   类别数: {len(LABELS)}")
print("\n   类别分布:")
for label_id, label_name in LABELS.items():
    count = len(train_df[train_df['label'] == label_id])
    print(f"   {label_id}. {label_name}: {count} 条")

# 转换为 Dataset 格式
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 2. 加载模型和分词器
print("\n🤖 步骤 2/6: 加载模型和分词器...")
model_name = "bert-base-chinese"
print(f"   模型: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(LABELS)
)

# 检测 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   设备: {device}")

# 3. 数据预处理
print("\n🔧 步骤 3/6: 数据预处理...")

def preprocess_function(examples):
    """对文本进行分词"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding=False  # 使用 DataCollator 动态padding
    )

# 对数据集进行预处理
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

print("   ✅ 数据预处理完成")

# 4. 定义评估函数
def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算精确率、召回率、F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 5. 训练配置
print("\n⚙️  步骤 4/6: 配置训练参数...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="no",  # 不保存中间checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none"  # 不使用wandb等
)

print(f"   学习率: {training_args.learning_rate}")
print(f"   批次大小: {training_args.per_device_train_batch_size}")
print(f"   训练轮数: {training_args.num_train_epochs}")
print(f"   输出目录: {OUTPUT_DIR}")

# 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. 创建 Trainer
print("\n🎯 步骤 5/6: 创建 Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 7. 开始训练
print("\n🚀 步骤 6/6: 开始训练...")
print("=" * 70)

train_result = trainer.train()

print("\n" + "=" * 70)
print("✅ 训练完成！")
print("=" * 70)

# 8. 评估模型
print("\n📊 最终评估...")
eval_results = trainer.evaluate()

print("\n评估结果:")
print(f"   准确率: {eval_results['eval_accuracy']:.4f}")
print(f"   精确率: {eval_results['eval_precision']:.4f}")
print(f"   召回率: {eval_results['eval_recall']:.4f}")
print(f"   F1分数: {eval_results['eval_f1']:.4f}")

# 9. 详细分类报告
print("\n📈 详细分类报告:")
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=1)

report = classification_report(
    test_df['label'],
    preds,
    target_names=[LABELS[i] for i in range(len(LABELS))],
    digits=4
)
print(report)

# 10. 保存模型
print("\n💾 保存模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 保存标签映射
import json
label_map_path = os.path.join(OUTPUT_DIR, 'label_map.json')
with open(label_map_path, 'w', encoding='utf-8') as f:
    json.dump(LABELS, f, ensure_ascii=False, indent=2)

print(f"   模型已保存到: {OUTPUT_DIR}")
print(f"   标签映射已保存到: {label_map_path}")

# 11. 测试推理
print("\n🧪 测试推理...")
print("=" * 70)

# 加载保存的模型
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model=OUTPUT_DIR,
    tokenizer=OUTPUT_DIR,
    device=0 if device == "cuda" else -1
)

# 测试样本
test_texts = [
    "苹果公司发布新款MacBook Pro，搭载M3芯片",
    "中国女篮战胜美国队，夺得世界杯冠军",
    "周杰伦新专辑发布，首日销量破百万",
    "A股大涨，沪指突破3500点",
    "北京今日有雨，气温下降",
    "教育部发布新规，规范校外培训"
]

print("\n测试样本预测:")
for text in test_texts:
    result = classifier(text)[0]
    label_id = int(result['label'].split('_')[-1])
    label_name = LABELS[label_id]
    score = result['score']
    print(f"\n文本: {text}")
    print(f"预测: {label_name} (置信度: {score:.4f})")

print("\n" + "=" * 70)
print("✨ 全部完成！")
print("=" * 70)
print("\n💡 使用说明:")
print(f"   1. 模型保存在: {OUTPUT_DIR}")
print("   2. 可以使用 pipeline 加载模型进行推理")
print("   3. 查看 label_map.json 了解类别映射")
print("\n📝 下一步:")
print("   - 运行 新闻分类测试.py 测试模型")
print("   - 或创建 Web 服务部署模型")
