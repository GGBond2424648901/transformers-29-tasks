#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 优化训练 - 通过更多轮数和更好的参数提升效果
"""

import os
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
import torch

print("=" * 70)
print("❓ BERT 优化训练 - 提升模型效果")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  设备: {device}")

# 1. 加载数据集
print("\n📦 加载完整数据集...")
try:
    dataset = load_dataset("clue", "cmrc2018", split="train")
    eval_dataset = load_dataset("clue", "cmrc2018", split="validation")
    print(f"✅ 训练集: {len(dataset)} 条")
    print(f"✅ 验证集: {len(eval_dataset)} 条")
except:
    from datasets import Dataset
    sample_data = {
        "context": ["北京是中华人民共和国的首都。"] * 1000,
        "question": ["北京是什么？"] * 1000,
        "answers": [{"text": ["中华人民共和国的首都"], "answer_start": [3]}] * 1000,
        "id": [f"sample_{i}" for i in range(1000)]
    }
    dataset = Dataset.from_dict(sample_data)
    eval_dataset = Dataset.from_dict({k: v[:100] for k, v in sample_data.items()})

# 2. 加载模型
print("\n📦 加载 BERT 模型...")
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
print(f"✅ 模型: {model_name}")

# 3. 数据预处理
print("\n📦 数据预处理...")

max_length = 512  # 增加到 512
doc_stride = 128

def preprocess_function(examples):
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

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
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

print("✅ 完成")

# 4. 优化的训练配置
print("\n📦 配置优化训练参数...")

training_args = TrainingArguments(
    output_dir="qa_model_output_bert_optimized",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,  # 稍微提高学习率
    per_device_train_batch_size=8 if device == "cuda" else 4,
    per_device_eval_batch_size=8 if device == "cuda" else 4,
    num_train_epochs=8,  # 增加到 8 轮
    weight_decay=0.01,
    warmup_steps=500,  # 添加预热
    logging_steps=100,
    fp16=device == "cuda",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    gradient_accumulation_steps=2,  # 梯度累积
)

print("✅ 优化参数:")
print(f"   训练轮数: 8 epochs（原来 3）")
print(f"   学习率: 3e-5（原来 2e-5）")
print(f"   序列长度: 512（原来 384）")
print(f"   预热步数: 500")
print(f"   梯度累积: 2 步")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=default_data_collator,
)

print("\n" + "=" * 70)
print("🚀 开始优化训练")
print("=" * 70)
print(f"📊 数据: {len(dataset)} 条训练，{len(eval_dataset)} 条验证")
print(f"⏱️  预计时间: 40-60 分钟 (GPU)")
print("=" * 70)

try:
    trainer.train()
    
    print("\n✅ 训练完成！")
    
    output_dir = "中文问答模型_BERT优化版"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 模型保存: ./{output_dir}")
    
    metrics = trainer.evaluate()
    print("\n最终评估:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 快速测试
    print("\n🧪 快速测试...")
    from transformers import pipeline
    
    qa = pipeline("question-answering", model=output_dir, device=0 if device == "cuda" else -1)
    
    test_cases = [
        ("北京是中华人民共和国的首都，是全国的政治中心。", "北京是什么？"),
        ("秦始皇连接和修缮战国长城，始有万里长城之称。", "谁修建了万里长城？"),
    ]
    
    for context, question in test_cases:
        result = qa(question=question, context=context)
        print(f"\n问题: {question}")
        print(f"答案: {result['answer']}")
        print(f"置信度: {result['score']:.2%}")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✨ 完成！")
print("=" * 70)
print(f"\n📁 模型位置: ./{output_dir if 'output_dir' in locals() else '中文问答模型_BERT优化版'}")
print("🧪 测试命令: python 中文问答测试_高级版.py")
