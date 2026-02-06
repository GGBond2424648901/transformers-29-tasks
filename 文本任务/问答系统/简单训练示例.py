#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问答系统简单训练示例
使用 SQuAD 格式数据训练中文问答模型
"""

import os

# 设置模型缓存路径
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from datasets import load_dataset

print("=" * 70)
print("❓ 问答系统训练示例")
print("=" * 70)

# 1. 加载数据集
print("\n📦 步骤 1: 加载数据集")
print("-" * 70)

# 使用中文问答数据集 CMRC2018
# 如果下载失败，会自动创建示例数据
try:
    print("正在下载 CMRC2018 中文数据集...")
    dataset = load_dataset("clue", "cmrc2018", split="train[:500]")  # 使用500条数据
    eval_dataset = load_dataset("clue", "cmrc2018", split="validation[:50]")
    print("✅ 数据集下载成功")
except Exception as e:
    print(f"⚠️  数据集下载失败: {e}")
    print("使用示例数据进行训练...")
    
    # 创建示例中文数据
    from datasets import Dataset
    
    sample_data = {
        "context": [
            "北京是中华人民共和国的首都，是全国的政治中心、文化中心。北京位于华北平原北部，背靠燕山，毗邻天津市和河北省。",
            "人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "长城是中国古代的军事防御工程，是一道高大、坚固而连绵不断的长垣，用以限隔敌骑的行动。长城不是一道单纯孤立的城墙，而是以城墙为主体，同大量的城、障、亭、标相结合的防御体系。"
        ] * 167,  # 重复以达到500条
        "question": [
            "北京是什么？",
            "人工智能是什么？",
            "长城的作用是什么？"
        ] * 167,
        "answers": [
            {"text": ["中华人民共和国的首都"], "answer_start": [3]},
            {"text": ["计算机科学的一个分支"], "answer_start": [9]},
            {"text": ["军事防御工程"], "answer_start": [7]}
        ] * 167,
        "id": [f"sample_{i}" for i in range(501)]
    }
    
    dataset = Dataset.from_dict(sample_data)
    eval_dataset = Dataset.from_dict({k: v[:50] for k, v in sample_data.items()})

print(f"✅ 训练集大小: {len(dataset)}")
print(f"✅ 验证集大小: {len(eval_dataset)}")

# 查看数据格式
print("\n📊 数据示例:")
example = dataset[0]
print(f"上下文: {example['context'][:100]}...")
print(f"问题: {example['question']}")
print(f"答案: {example['answers']}")

# 2. 加载模型和分词器
print("\n📦 步骤 2: 加载模型")
print("-" * 70)

model_name = "bert-base-chinese"  # 中文 BERT 模型
# 其他中文模型选择:
# "hfl/chinese-roberta-wwm-ext" - 中文 RoBERTa（效果更好）
# "hfl/chinese-bert-wwm-ext" - 中文 BERT WWM

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

print(f"✅ 模型加载成功: {model_name}")

# 3. 数据预处理
print("\n📦 步骤 3: 数据预处理")
print("-" * 70)

max_length = 384
doc_stride = 128

def preprocess_function(examples):
    """
    预处理问答数据
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # 找到上下文的开始和结束
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # 如果答案不在上下文中，标记为 (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # 否则找到答案的 token 位置
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# 处理数据集
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
)

tokenized_eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

print(f"✅ 数据预处理完成")

# 4. 训练配置
print("\n📦 步骤 4: 配置训练参数")
print("-" * 70)

training_args = TrainingArguments(
    output_dir="./qa_model_output",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=50,
    fp16=True,  # 使用混合精度训练（需要 GPU）
)

print("✅ 训练参数:")
print(f"   批次大小: {training_args.per_device_train_batch_size}")
print(f"   学习率: {training_args.learning_rate}")
print(f"   训练轮数: {training_args.num_train_epochs}")

# 5. 创建 Trainer
print("\n📦 步骤 5: 创建 Trainer")
print("-" * 70)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=default_data_collator,
)

print("✅ Trainer 创建成功")

# 6. 开始训练
print("\n" + "=" * 70)
print("🚀 开始训练")
print("=" * 70)

try:
    trainer.train()
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    
    # 7. 保存模型
    print("\n📦 保存模型...")
    output_dir = "中文问答模型"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 模型已保存到: ./{output_dir}")
    
    # 8. 评估模型
    print("\n📊 评估模型...")
    metrics = trainer.evaluate()
    
    print("\n评估结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

except Exception as e:
    print(f"\n❌ 训练出错: {e}")
    print("\n💡 提示:")
    print("1. 确保已安装所有依赖: pip install -r requirements.txt")
    print("2. 如果内存不足，减小 batch_size")
    print("3. 如果没有 GPU，移除 fp16=True 参数")

print("\n" + "=" * 70)
print("✨ 示例完成！")
print("=" * 70)
