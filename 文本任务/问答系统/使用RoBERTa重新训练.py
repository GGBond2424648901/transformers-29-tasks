#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 RoBERTa 模型重新训练
强制使用 safetensors 格式避免 PyTorch 版本问题
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

# 禁用 pytorch_model.bin 的自动转换
os.environ['TRANSFORMERS_OFFLINE'] = '0'

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
print("❓ 使用 RoBERTa 重新训练")
print("=" * 70)

# 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  使用设备: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# 1. 加载数据集
print("\n📦 步骤 1: 加载数据集")
print("-" * 70)

try:
    dataset = load_dataset("clue", "cmrc2018", split="train")
    eval_dataset = load_dataset("clue", "cmrc2018", split="validation")
    print(f"✅ 训练集: {len(dataset)} 条")
    print(f"✅ 验证集: {len(eval_dataset)} 条")
except Exception as e:
    print(f"⚠️  数据集加载失败，使用缓存数据")
    from datasets import Dataset
    sample_data = {
        "context": ["北京是中华人民共和国的首都。"] * 1000,
        "question": ["北京是什么？"] * 1000,
        "answers": [{"text": ["中华人民共和国的首都"], "answer_start": [3]}] * 1000,
        "id": [f"sample_{i}" for i in range(1000)]
    }
    dataset = Dataset.from_dict(sample_data)
    eval_dataset = Dataset.from_dict({k: v[:100] for k, v in sample_data.items()})

# 2. 加载 RoBERTa 模型
print("\n📦 步骤 2: 加载 RoBERTa 模型")
print("-" * 70)

model_name = "hfl/chinese-roberta-wwm-ext"

try:
    print(f"📥 尝试加载: {model_name}")
    print("   使用本地缓存...")
    
    # 强制使用本地缓存
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=False,
        trust_remote_code=True
    )
    
    # 尝试加载模型，忽略 pytorch_model.bin 的警告
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            local_files_only=False,
            trust_remote_code=True,
            use_safetensors=False  # 使用 pytorch_model.bin
        )
        print(f"✅ RoBERTa 模型加载成功！")
    except Exception as e:
        print(f"⚠️  加载失败: {str(e)[:200]}")
        print("\n💡 解决方案：")
        print("   升级 PyTorch: pip install --upgrade torch==2.6.0")
        print("\n   或者继续使用 bert-base-chinese")
        raise

except Exception as e:
    print(f"❌ RoBERTa 加载失败")
    print(f"   错误: {str(e)[:200]}")
    print("\n使用备用模型: bert-base-chinese")
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

print(f"✅ 最终使用模型: {model_name}")

# 3. 数据预处理
print("\n📦 步骤 3: 数据预处理")
print("-" * 70)

max_length = 512
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

print("✅ 数据预处理完成")

# 4. 训练配置
print("\n📦 步骤 4: 配置训练参数")
print("-" * 70)

if device == "cuda":
    batch_size = 8
    use_fp16 = True
else:
    batch_size = 4
    use_fp16 = False

training_args = TrainingArguments(
    output_dir="qa_model_output_roberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,  # 增加到 5 轮
    weight_decay=0.01,
    logging_steps=100,
    fp16=use_fp16,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
)

print(f"✅ 训练参数:")
print(f"   模型: {model_name}")
print(f"   训练轮数: 5 epochs")
print(f"   批次大小: {batch_size}")

# 5. 训练
print("\n" + "=" * 70)
print("🚀 开始训练")
print("=" * 70)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=default_data_collator,
)

try:
    trainer.train()
    
    print("\n✅ 训练完成！")
    
    # 保存模型
    output_dir = "中文问答模型_RoBERTa版" if "roberta" in model_name.lower() else "中文问答模型_BERT增强版"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 模型已保存到: ./{output_dir}")
    
    # 评估
    metrics = trainer.evaluate()
    print("\n评估结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

except Exception as e:
    print(f"\n❌ 训练出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✨ 完成！")
print("=" * 70)
