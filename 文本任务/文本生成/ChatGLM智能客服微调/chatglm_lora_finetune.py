#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM-6B LoRA 微调脚本 - 智能客服版本
使用 LoRA 技术微调 ChatGLM-6B，打造专属客服助手
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import json
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

print("=" * 70)
print("🤖 Qwen2.5-1.5B LoRA 微调 - 智能客服")
print("=" * 70)

# ============================================================================
# 1. 配置参数
# ============================================================================

@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Qwen2.5-1.5B 模型路径 (3GB, 中文支持好, 兼容性强)"}
    )

@dataclass
class DataArguments:
    """数据参数"""
    train_file: str = field(
        default="data/train.json",
        metadata={"help": "训练数据文件"}
    )
    validation_file: str = field(
        default="data/dev.json",
        metadata={"help": "验证数据文件"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )

@dataclass
class LoraArguments:
    """LoRA 参数"""
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA 秩"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )

# ============================================================================
# 2. 加载数据
# ============================================================================

def load_data(file_path):
    """加载 JSON 格式的训练数据"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建绝对路径
    abs_file_path = os.path.join(script_dir, file_path)
    
    print(f"\n📥 加载数据: {abs_file_path}")
    
    with open(abs_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 加载了 {len(data)} 条数据")
    
    # 转换为 Dataset 格式
    dataset = Dataset.from_list(data)
    return dataset

# ============================================================================
# 3. 数据预处理
# ============================================================================

def preprocess_function(examples, tokenizer, max_length=512):
    """
    数据预处理函数
    将 instruction + input + output 转换为模型输入格式
    """
    model_inputs = {"input_ids": [], "labels": []}
    
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        output_text = examples["output"][i]
        
        # 构建提示词
        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction
        
        # 构建完整对话
        # ChatGLM 格式：[Round 1]\n\n问：{prompt}\n\n答：{output}
        a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=max_length)
        b_ids = tokenizer.encode(text=output_text, add_special_tokens=False, truncation=True, max_length=max_length)
        
        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
        
        model_inputs["input_ids"].append(input_ids)
        model_inputs["labels"].append(labels)
    
    return model_inputs

# ============================================================================
# 4. 主训练函数
# ============================================================================

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 参数设置
    model_args = ModelArguments()
    data_args = DataArguments()
    lora_args = LoraArguments()
    
    # 输出目录（使用绝对路径）
    output_dir = os.path.join(script_dir, "output/chatglm-customer-lora")
    
    print("\n" + "=" * 70)
    print("⚙️  训练配置")
    print("=" * 70)
    print(f"模型: {model_args.model_name_or_path}")
    print(f"训练数据: {data_args.train_file}")
    print(f"验证数据: {data_args.validation_file}")
    print(f"LoRA Rank: {lora_args.lora_rank}")
    print(f"LoRA Alpha: {lora_args.lora_alpha}")
    print(f"输出目录: {output_dir}")
    
    # ========================================================================
    # 加载 Tokenizer 和模型
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("📦 加载模型")
    print("=" * 70)
    
    try:
        print(f"正在加载 tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
        
        # 修复 ChatGLM tokenizer 兼容性问题
        if not hasattr(tokenizer, 'vocab_size'):
            tokenizer.vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 130528
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2
        
        print("✅ Tokenizer 加载成功")
        
        print(f"\n正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✅ 模型加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n💡 提示：")
        print("1. 首次运行会自动下载 ChatGLM-6B（约 12GB）")
        print("2. 请确保网络连接正常")
        print("3. 或手动下载模型到预训练模型下载处")
        return
    
    # ========================================================================
    # 配置 LoRA
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("🔧 配置 LoRA")
    print("=" * 70)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Qwen2 的注意力层
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # ========================================================================
    # 加载和预处理数据
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("📊 准备数据")
    print("=" * 70)
    
    train_dataset = load_data(data_args.train_file)
    eval_dataset = load_data(data_args.validation_file)
    
    # 预处理
    print("\n处理训练数据...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    print("处理验证数据...")
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args.max_length),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 条")
    print(f"✅ 验证集: {len(eval_dataset)} 条")
    
    # ========================================================================
    # 训练参数
    # ========================================================================
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # ========================================================================
    # 开始训练
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("🚀 开始训练")
    print("=" * 70)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    try:
        trainer.train()
        
        print("\n" + "=" * 70)
        print("💾 保存模型")
        print("=" * 70)
        
        # 保存 LoRA 权重
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ 模型已保存到: {output_dir}")
        
        print("\n" + "=" * 70)
        print("✨ 训练完成！")
        print("=" * 70)
        print("\n下一步：")
        print("1. 运行 test_model.py 测试模型")
        print("2. 运行 启动客服系统.bat 启动 Web 服务")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        print("\n💡 可能的原因：")
        print("1. 显存不足 - 尝试减小 batch_size")
        print("2. 数据格式错误 - 检查 JSON 文件")
        print("3. 模型加载失败 - 检查网络连接")

if __name__ == "__main__":
    main()
