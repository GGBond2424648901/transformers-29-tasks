#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问答系统高级训练示例
使用完整 CMRC2018 数据集 + 更好的中文模型
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
import torch

print("=" * 70)
print("❓ 问答系统高级训练示例")
print("=" * 70)

# 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  使用设备: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 1. 加载数据集
print("\n📦 步骤 1: 加载完整数据集")
print("-" * 70)

try:
    print("正在下载 CMRC2018 完整数据集...")
    print("⚠️  首次下载可能需要几分钟，请耐心等待...")
    
    dataset = load_dataset("clue", "cmrc2018", split="train")
    eval_dataset = load_dataset("clue", "cmrc2018", split="validation")
    
    print("✅ 数据集下载成功")
    print(f"✅ 训练集大小: {len(dataset)} 条")
    print(f"✅ 验证集大小: {len(eval_dataset)} 条")
    
except Exception as e:
    print(f"⚠️  数据集下载失败: {e}")
    print("使用较小的数据集进行训练...")
    
    # 如果下载失败，使用部分数据
    from datasets import Dataset
    
    sample_data = {
        "context": [
            "北京是中华人民共和国的首都，是全国的政治中心、文化中心。北京位于华北平原北部，背靠燕山，毗邻天津市和河北省。",
            "人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "长城是中国古代的军事防御工程，是一道高大、坚固而连绵不断的长垣，用以限隔敌骑的行动。"
        ] * 334,
        "question": [
            "北京是什么？",
            "人工智能是什么？",
            "长城的作用是什么？"
        ] * 334,
        "answers": [
            {"text": ["中华人民共和国的首都"], "answer_start": [3]},
            {"text": ["计算机科学的一个分支"], "answer_start": [9]},
            {"text": ["军事防御工程"], "answer_start": [7]}
        ] * 334,
        "id": [f"sample_{i}" for i in range(1002)]
    }
    
    dataset = Dataset.from_dict(sample_data)
    eval_dataset = Dataset.from_dict({k: v[:100] for k, v in sample_data.items()})
    
    print(f"✅ 训练集大小: {len(dataset)} 条")
    print(f"✅ 验证集大小: {len(eval_dataset)} 条")

# 查看数据格式
print("\n📊 数据示例:")
example = dataset[0]
print(f"上下文: {example['context'][:100]}...")
print(f"问题: {example['question']}")
print(f"答案: {example['answers']}")

# 2. 加载模型和分词器
print("\n📦 步骤 2: 加载模型")
print("-" * 70)

# 使用中文模型
# 优先尝试更好的 RoBERTa 模型，失败则使用 BERT
model_options = [
    ("hfl/chinese-roberta-wwm-ext", "中文 RoBERTa（推荐）"),
    ("bert-base-chinese", "中文 BERT（备用）")
]

model_name = None
for model_id, model_desc in model_options:
    try:
        print(f"📥 尝试加载: {model_desc}")
        print(f"   模型ID: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        
        model_name = model_id
        print(f"✅ 模型加载成功: {model_desc}")
        break
    except Exception as e:
        print(f"⚠️  加载失败: {str(e)[:100]}...")
        print(f"   尝试下一个模型...\n")
        continue

if model_name is None:
    raise RuntimeError("所有模型都加载失败，请检查网络连接")

# 3. 数据预处理
print("\n📦 步骤 3: 数据预处理")
print("-" * 70)

max_length = 512  # 增加到 512
doc_stride = 128

def preprocess_function(examples):
    """预处理问答数据"""
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

print("正在处理数据集...")
# 处理数据集
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="处理训练集"
)

tokenized_eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="处理验证集"
)

print(f"✅ 数据预处理完成")

# 4. 训练配置
print("\n📦 步骤 4: 配置训练参数")
print("-" * 70)

# 根据设备调整参数
if device == "cuda":
    batch_size = 8
    use_fp16 = True
    print("✅ 使用 GPU 训练（FP16 混合精度）")
else:
    batch_size = 4
    use_fp16 = False
    print("⚠️  使用 CPU 训练（训练时间会较长）")

training_args = TrainingArguments(
    output_dir="qa_model_output_advanced",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # 稍微降低学习率
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,  # 增加到 3 轮
    weight_decay=0.01,
    logging_steps=100,
    fp16=use_fp16,
    load_best_model_at_end=True,  # 加载最佳模型
    metric_for_best_model="eval_loss",
    save_total_limit=2,  # 只保留最好的 2 个检查点
)

print("✅ 训练参数:")
print(f"   模型: {model_name}")
print(f"   批次大小: {training_args.per_device_train_batch_size}")
print(f"   学习率: {training_args.learning_rate}")
print(f"   训练轮数: {training_args.num_train_epochs}")
print(f"   最大序列长度: {max_length}")
print(f"   混合精度: {use_fp16}")

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
print(f"📊 训练数据: {len(dataset)} 条")
print(f"📊 验证数据: {len(eval_dataset)} 条")
print(f"⏱️  预计时间: {'15-30 分钟' if device == 'cuda' else '1-2 小时'}")
print("=" * 70)

try:
    trainer.train()
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    
    # 7. 保存模型
    print("\n📦 保存模型...")
    output_dir = "中文问答模型_高级版"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 模型已保存到: ./{output_dir}")
    
    # 8. 评估模型
    print("\n📊 最终评估...")
    metrics = trainer.evaluate()
    
    print("\n评估结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 9. 快速测试
    print("\n" + "=" * 70)
    print("🧪 快速测试")
    print("=" * 70)
    
    from transformers import pipeline
    
    qa_pipeline = pipeline(
        "question-answering",
        model=output_dir,
        device=0 if device == "cuda" else -1
    )
    
    test_context = "北京是中华人民共和国的首都，是全国的政治中心、文化中心。北京位于华北平原北部。"
    test_question = "北京是什么？"
    
    result = qa_pipeline(question=test_question, context=test_context)
    
    print(f"\n测试问题: {test_question}")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['score']:.2%}")

except Exception as e:
    print(f"\n❌ 训练出错: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 提示:")
    print("1. 确保已安装所有依赖")
    print("2. 如果内存不足，减小 batch_size")
    print("3. 如果没有 GPU，训练时间会很长")

print("\n" + "=" * 70)
print("✨ 示例完成！")
print("=" * 70)
print(f"\n📁 模型保存位置: ./{output_dir if 'output_dir' in locals() else '中文问答模型_高级版'}")
print("🧪 运行测试: python 中文问答测试_高级版.py")
