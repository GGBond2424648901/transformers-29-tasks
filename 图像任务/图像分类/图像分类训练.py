#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分类训练 - 使用ViT模型进行图像分类
支持自动创建示例数据集
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
MODEL_DIR = os.path.join(CURRENT_DIR, 'trained_model')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'training_results')

# 创建目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("🖼️ 图像分类训练")
print("=" * 70)

# 定义类别
CATEGORIES = ['猫', '狗', '鸟', '鱼', '马']
id2label = {i: label for i, label in enumerate(CATEGORIES)}
label2id = {label: i for i, label in enumerate(CATEGORIES)}



class ImageClassificationDataset(Dataset):
    """图像分类数据集"""
    
    def __init__(self, data_dir, processor):
        self.data_dir = data_dir
        self.processor = processor
        self.images = []
        self.labels = []
        
        # 加载所有图像
        for label_idx, category in enumerate(CATEGORIES):
            category_dir = os.path.join(data_dir, category)
            if not os.path.exists(category_dir):
                continue
            
            for img_name in os.listdir(category_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(category_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 预处理
        encoding = self.processor(images=image, return_tensors="pt")
        
        # 移除batch维度
        pixel_values = encoding['pixel_values'].squeeze()
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def plot_training_results(log_history, save_path):
    """绘制训练结果图"""
    print("\n📊 生成训练结果图...")
    
    # 提取训练和验证指标
    train_loss = []
    eval_loss = []
    eval_accuracy = []
    eval_f1 = []
    steps = []
    eval_steps = []
    
    for log in log_history:
        if 'loss' in log:
            train_loss.append(log['loss'])
            steps.append(log.get('step', len(train_loss)))
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])
            eval_accuracy.append(log.get('eval_accuracy', 0))
            eval_f1.append(log.get('eval_f1', 0))
            eval_steps.append(log.get('step', len(eval_loss)))
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('图像分类训练结果', fontsize=16, fontweight='bold')
    
    # 训练损失
    if train_loss:
        axes[0, 0].plot(steps, train_loss, 'b-', linewidth=2, label='训练损失')
        axes[0, 0].set_xlabel('训练步数')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].set_title('训练损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 验证损失
    if eval_loss:
        axes[0, 1].plot(eval_steps, eval_loss, 'r-', linewidth=2, label='验证损失')
        axes[0, 1].set_xlabel('训练步数')
        axes[0, 1].set_ylabel('损失值')
        axes[0, 1].set_title('验证损失曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 准确率
    if eval_accuracy:
        axes[1, 0].plot(eval_steps, eval_accuracy, 'g-', linewidth=2, label='准确率')
        axes[1, 0].set_xlabel('训练步数')
        axes[1, 0].set_ylabel('准确率')
        axes[1, 0].set_title('验证准确率曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # F1分数
    if eval_f1:
        axes[1, 1].plot(eval_steps, eval_f1, 'm-', linewidth=2, label='F1分数')
        axes[1, 1].set_xlabel('训练步数')
        axes[1, 1].set_ylabel('F1分数')
        axes[1, 1].set_title('验证F1分数曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练结果图已保存: {save_path}")
    plt.close()

def main():
    # 检查数据集是否存在
    train_dir = os.path.join(DATA_DIR, 'train')
    if not os.path.exists(train_dir):
        print(f"\n❌ 错误: 数据集不存在！")
        print(f"   请确保数据集位于: {train_dir}")
        print(f"   数据集应包含以下类别文件夹: {', '.join(CATEGORIES)}")
        return
    
    print("\n🔧 加载预训练模型...")
    model_name = "google/vit-base-patch16-224"
    
    # 加载处理器和模型
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(CATEGORIES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    print("✅ 模型加载完成")
    
    # 创建数据集
    print("\n📚 准备数据集...")
    train_dataset = ImageClassificationDataset(
        os.path.join(DATA_DIR, 'train'),
        processor
    )
    val_dataset = ImageClassificationDataset(
        os.path.join(DATA_DIR, 'val'),
        processor
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(RESULTS_DIR, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",  # 不保存中间checkpoint
        load_best_model_at_end=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("\n🚀 开始训练...")
    print("=" * 70)
    
    train_result = trainer.train()
    
    # 保存模型
    print("\n💾 保存模型...")
    try:
        trainer.save_model(MODEL_DIR)
        processor.save_pretrained(MODEL_DIR)
        print(f"✅ 模型已保存到: {MODEL_DIR}")
    except Exception as e:
        print(f"⚠️ 保存模型时出错: {e}")
        # 尝试手动保存
        model.save_pretrained(MODEL_DIR)
        processor.save_pretrained(MODEL_DIR)
        print(f"✅ 模型已手动保存到: {MODEL_DIR}")
    
    # 保存训练信息
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 最终评估
    print("\n📊 最终评估...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # 绘制训练结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(RESULTS_DIR, f'training_results_{timestamp}.png')
    plot_training_results(trainer.state.log_history, plot_path)
    
    # 保存训练摘要
    summary = {
        '训练时间': timestamp,
        '模型': model_name,
        '类别数': len(CATEGORIES),
        '类别': CATEGORIES,
        '训练样本数': len(train_dataset),
        '验证样本数': len(val_dataset),
        '训练轮数': training_args.num_train_epochs,
        '最终指标': {
            '准确率': f"{eval_metrics.get('eval_accuracy', 0):.4f}",
            'F1分数': f"{eval_metrics.get('eval_f1', 0):.4f}",
            '精确率': f"{eval_metrics.get('eval_precision', 0):.4f}",
            '召回率': f"{eval_metrics.get('eval_recall', 0):.4f}",
        }
    }
    
    summary_path = os.path.join(RESULTS_DIR, f'training_summary_{timestamp}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"\n📁 模型保存位置: {MODEL_DIR}")
    print(f"📊 训练结果图: {plot_path}")
    print(f"📄 训练摘要: {summary_path}")
    print(f"\n🎯 最终准确率: {eval_metrics.get('eval_accuracy', 0):.2%}")
    print(f"🎯 最终F1分数: {eval_metrics.get('eval_f1', 0):.4f}")

if __name__ == '__main__':
    main()
