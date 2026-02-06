#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
姿态估计训练 - LoRA微调版本
使用LoRA大幅减小模型大小
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    ViTImageProcessor,
    ViTModel,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
MODEL_DIR = os.path.join(CURRENT_DIR, 'trained_model_lora')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'training_results_lora')

# 创建目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("🤸 姿态估计训练 - LoRA微调版本")
print("=" * 70)

# 定义关键点
KEYPOINTS = ['头部', '左手', '右手', '左脚', '右脚']
NUM_KEYPOINTS = len(KEYPOINTS)

class PoseEstimationModel(nn.Module):
    """姿态估计模型（基于ViT + LoRA）"""
    
    def __init__(self, vit_model, num_keypoints=5):
        super().__init__()
        self.vit = vit_model
        self.num_keypoints = num_keypoints
        
        # 关键点回归头
        hidden_size = self.vit.config.hidden_size
        self.keypoint_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_keypoints * 2)
        )
    
    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        
        keypoints = self.keypoint_head(pooled_output)
        keypoints = keypoints.view(-1, self.num_keypoints, 2)
        
        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(keypoints, labels)
        
        return {'loss': loss, 'keypoints': keypoints}

class PoseDataset(Dataset):
    """姿态估计数据集"""
    
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
        
        image = Image.open(img_path).convert('RGB')
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding['pixel_values'].squeeze()
        
        keypoints = torch.tensor(annotation['keypoints'], dtype=torch.float32)
        
        return {
            'pixel_values': pixel_values,
            'labels': keypoints
        }

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    pixel_error = mae * 224
    
    return {
        'mse': mse,
        'mae': mae,
        'pixel_error': pixel_error
    }

class PoseTrainer(Trainer):
    """自定义姿态估计训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs['loss']
            keypoints = outputs['keypoints']
        
        return (loss, keypoints, labels)

def plot_training_results(log_history, save_path):
    """绘制训练结果图"""
    print("\n📊 生成训练结果图...")
    
    train_loss = []
    eval_loss = []
    eval_mae = []
    eval_pixel_error = []
    steps = []
    eval_steps = []
    
    for log in log_history:
        if 'loss' in log:
            train_loss.append(log['loss'])
            steps.append(log.get('step', len(train_loss)))
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])
            eval_mae.append(log.get('eval_mae', 0))
            eval_pixel_error.append(log.get('eval_pixel_error', 0))
            eval_steps.append(log.get('step', len(eval_loss)))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('姿态估计训练结果 (LoRA)', fontsize=16, fontweight='bold')
    
    if train_loss:
        axes[0, 0].plot(steps, train_loss, 'b-', linewidth=2, label='训练损失')
        axes[0, 0].set_xlabel('训练步数')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].set_title('训练损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    if eval_loss:
        axes[0, 1].plot(eval_steps, eval_loss, 'r-', linewidth=2, label='验证损失')
        axes[0, 1].set_xlabel('训练步数')
        axes[0, 1].set_ylabel('损失值')
        axes[0, 1].set_title('验证损失曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    if eval_mae:
        axes[1, 0].plot(eval_steps, eval_mae, 'g-', linewidth=2, label='平均绝对误差')
        axes[1, 0].set_xlabel('训练步数')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('平均绝对误差曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if eval_pixel_error:
        axes[1, 1].plot(eval_steps, eval_pixel_error, 'm-', linewidth=2, label='像素误差')
        axes[1, 1].set_xlabel('训练步数')
        axes[1, 1].set_ylabel('像素误差')
        axes[1, 1].set_title('关键点像素误差曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练结果图已保存: {save_path}")
    plt.close()

def main():
    # 检查数据集
    images_dir = os.path.join(DATA_DIR, 'images')
    annotations_dir = os.path.join(DATA_DIR, 'annotations')
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"\n❌ 错误: 数据集不存在！")
        print(f"   请确保数据集位于:")
        print(f"   - 图像: {images_dir}")
        print(f"   - 标注: {annotations_dir}")
        return
    
    print("\n🔧 加载预训练模型...")
    model_name = "google/vit-base-patch16-224"
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    vit_model = ViTModel.from_pretrained(model_name)
    
    print("✅ 基础模型加载完成")
    
    # 配置LoRA
    print("\n🎯 配置LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    
    # 应用LoRA到ViT
    vit_model = get_peft_model(vit_model, lora_config)
    vit_model.print_trainable_parameters()
    
    # 创建姿态估计模型
    model = PoseEstimationModel(vit_model, num_keypoints=NUM_KEYPOINTS)
    
    print("✅ LoRA配置完成")
    
    # 创建数据集
    print("\n📚 准备数据集...")
    train_dataset = PoseDataset(DATA_DIR, processor, 'train')
    val_dataset = PoseDataset(DATA_DIR, processor, 'val')
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=30,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,  # LoRA用更高的学习率
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(RESULTS_DIR, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
    )
    
    # 创建训练器
    trainer = PoseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("\n🚀 开始LoRA微调...")
    print("=" * 70)
    
    train_result = trainer.train()
    
    # 保存模型
    print("\n💾 保存LoRA模型...")
    try:
        # 保存LoRA权重
        model.vit.save_pretrained(os.path.join(MODEL_DIR, 'vit_lora'))
        # 保存完整模型权重
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'pytorch_model.bin'))
        processor.save_pretrained(MODEL_DIR)
        
        # 保存模型配置
        model_config = {
            'model_type': 'pose_estimation_lora',
            'base_model': model_name,
            'num_keypoints': NUM_KEYPOINTS,
            'keypoints': KEYPOINTS,
            'lora_config': {
                'r': lora_config.r,
                'lora_alpha': lora_config.lora_alpha,
                'target_modules': lora_config.target_modules,
            }
        }
        with open(os.path.join(MODEL_DIR, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(model_config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ LoRA模型已保存到: {MODEL_DIR}")
        
        # 检查模型大小
        import glob
        model_files = glob.glob(os.path.join(MODEL_DIR, '**/*.bin'), recursive=True) + \
                     glob.glob(os.path.join(MODEL_DIR, '**/*.safetensors'), recursive=True)
        if model_files:
            total_size = sum(os.path.getsize(f) for f in model_files) / (1024 * 1024)
            print(f"📦 LoRA模型大小: {total_size:.2f} MB (原模型: ~330 MB)")
    except Exception as e:
        print(f"⚠️ 保存模型时出错: {e}")
    
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
        '微调方法': 'LoRA',
        'LoRA配置': {
            'r': lora_config.r,
            'lora_alpha': lora_config.lora_alpha,
            'target_modules': lora_config.target_modules,
        },
        '关键点数': NUM_KEYPOINTS,
        '关键点': KEYPOINTS,
        '训练样本数': len(train_dataset),
        '验证样本数': len(val_dataset),
        '训练轮数': training_args.num_train_epochs,
        '最终指标': {
            'MAE': f"{eval_metrics.get('eval_mae', 0):.6f}",
            'MSE': f"{eval_metrics.get('eval_mse', 0):.6f}",
            '像素误差': f"{eval_metrics.get('eval_pixel_error', 0):.2f}像素",
        }
    }
    
    summary_path = os.path.join(RESULTS_DIR, f'training_summary_{timestamp}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ LoRA微调完成！")
    print("=" * 70)
    print(f"\n📁 模型保存位置: {MODEL_DIR}")
    print(f"📊 训练结果图: {plot_path}")
    print(f"📄 训练摘要: {summary_path}")
    print(f"\n🎯 最终像素误差: {eval_metrics.get('eval_pixel_error', 0):.2f} 像素")
    print(f"🎯 最终MAE: {eval_metrics.get('eval_mae', 0):.6f}")
    print(f"\n💡 LoRA优势: 模型大小从~330MB大幅减少！")

if __name__ == '__main__':
    main()
