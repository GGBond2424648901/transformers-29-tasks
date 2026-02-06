#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测训练 - 使用DETR模型进行目标检测
支持自动创建示例数据集
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import torch
from torch.utils.data import Dataset
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    TrainingArguments,
    Trainer
)
from PIL import Image, ImageDraw
import numpy as np
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
print("🎯 目标检测训练")
print("=" * 70)

# 定义类别（COCO格式，0是背景）
CATEGORIES = ['背景', '圆形', '方形', '三角形']
id2label = {i: label for i, label in enumerate(CATEGORIES)}
label2id = {label: i for i, label in enumerate(CATEGORIES)}



class DetectionDataset(Dataset):
    """目标检测数据集"""
    
    def __init__(self, data_dir, processor, split='train'):
        self.data_dir = data_dir
        self.processor = processor
        self.split = split
        
        images_dir = os.path.join(data_dir, 'images')
        annotations_dir = os.path.join(data_dir, 'annotations')
        
        self.samples = []
        for ann_file in os.listdir(annotations_dir):
            if ann_file.startswith(split) and ann_file.endswith('.json'):
                ann_path = os.path.join(annotations_dir, ann_file)
                with open(ann_path, 'r') as f:
                    annotation = json.load(f)
                
                img_path = os.path.join(images_dir, annotation['file_name'])
                if os.path.exists(img_path):
                    self.samples.append((img_path, annotation))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, annotation = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # 准备COCO格式标注
        boxes = []
        labels = []
        for ann in annotation['annotations']:
            # COCO格式: [x_min, y_min, width, height]
            x, y, width, height = ann['bbox']
            # 转换为归一化的 [x_center, y_center, width, height]
            x_center = (x + width / 2) / w
            y_center = (y + height / 2) / h
            norm_width = width / w
            norm_height = height / h
            boxes.append([x_center, y_center, norm_width, norm_height])
            labels.append(ann['category_id'])
        
        # 构建COCO格式的target
        target = {
            'image_id': annotation['image_id'],
            'annotations': [
                {
                    'image_id': annotation['image_id'],
                    'category_id': label,
                    'bbox': box,
                    'area': box[2] * box[3] * w * h,  # 实际面积
                    'iscrowd': 0
                }
                for box, label in zip(boxes, labels)
            ]
        }
        
        # 预处理
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        # 移除batch维度
        pixel_values = encoding['pixel_values'].squeeze(0)
        labels = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                 for k, v in encoding['labels'][0].items()}
        
        return {'pixel_values': pixel_values, 'labels': labels}

def collate_fn(batch):
    """自定义批处理函数"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['labels'] for item in batch]
    return {'pixel_values': pixel_values, 'labels': labels}

def plot_training_results(log_history, save_path):
    """绘制训练结果图"""
    print("\n📊 生成训练结果图...")
    
    train_loss = []
    steps = []
    
    for log in log_history:
        if 'loss' in log:
            train_loss.append(log['loss'])
            steps.append(log.get('step', len(train_loss)))
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('目标检测训练结果', fontsize=16, fontweight='bold')
    
    if train_loss:
        ax.plot(steps, train_loss, 'b-', linewidth=2, label='训练损失')
        ax.set_xlabel('训练步数')
        ax.set_ylabel('损失值')
        ax.set_title('训练损失曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练结果图已保存: {save_path}")
    plt.close()

def main():
    # 检查数据集是否存在
    images_dir = os.path.join(DATA_DIR, 'images')
    annotations_dir = os.path.join(DATA_DIR, 'annotations')
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"\n❌ 错误: 数据集不存在！")
        print(f"   请确保数据集位于:")
        print(f"   - 图像: {images_dir}")
        print(f"   - 标注: {annotations_dir}")
        return
    
    print("\n🔧 加载预训练模型...")
    model_name = "facebook/detr-resnet-50"
    
    # 加载处理器和模型
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=len(CATEGORIES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    print("✅ 模型加载完成")
    
    # 创建数据集
    print("\n📚 准备数据集...")
    train_dataset = DetectionDataset(DATA_DIR, processor, 'train')
    val_dataset = DetectionDataset(DATA_DIR, processor, 'val')
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=20,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(RESULTS_DIR, 'logs'),
        logging_steps=10,
        save_strategy="no",  # 不保存中间checkpoint
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
        data_collator=collate_fn,
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
        '最终训练损失': f"{metrics.get('train_loss', 0):.4f}",
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

if __name__ == '__main__':
    main()
