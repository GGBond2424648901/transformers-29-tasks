#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分类训练 - LoRA微调版本
使用LoRA大幅减小模型大小（从330MB到~10MB）
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

import torch
from torch.utils.data import Dataset
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
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
print("🖼️ 图像分类训练 - LoRA微调版本")
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
        
        image = Image.open(img_path).convert('RGB')
        encoding = self.processor(images=image, return_tensors="pt")
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

def plot_training_results(plot_data, save_path):
    """绘制训练结果图"""
    print("\n📊 生成训练结果图...")
    
    log_history = plot_data['log_history']
    accuracy = plot_data['accuracy']
    precision = plot_data['precision']
    recall = plot_data['recall']
    f1 = plot_data['f1']
    conf_matrix = plot_data['conf_matrix']
    per_class_metrics = plot_data['per_class_metrics']
    
    # 提取训练损失
    train_loss = []
    steps = []
    
    for log in log_history:
        if 'loss' in log and 'eval_loss' not in log:
            train_loss.append(log['loss'])
            steps.append(log.get('step', len(train_loss)))
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('图像分类 LoRA 训练结果', fontsize=16, fontweight='bold')
    
    # 1. 训练损失曲线
    if train_loss:
        axes[0, 0].plot(steps, train_loss, 'b-', linewidth=2, label='训练损失', marker='o', markersize=4)
        axes[0, 0].set_xlabel('训练步数', fontsize=11)
        axes[0, 0].set_ylabel('损失值', fontsize=11)
        axes[0, 0].set_title('训练损失曲线', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加最终损失值标注
        if len(train_loss) > 0:
            final_loss = train_loss[-1]
            final_step = steps[-1]
            axes[0, 0].annotate(f'{final_loss:.4f}', 
                               xy=(final_step, final_loss), 
                               xytext=(10, 10), 
                               textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               fontsize=9)
    
    # 2. 整体指标柱状图
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    metrics_values = [accuracy, precision, recall, f1]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = axes[0, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('分数', fontsize=11)
    axes[0, 1].set_title('分类性能指标', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 混淆矩阵
    import seaborn as sns
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CATEGORIES, yticklabels=CATEGORIES,
                ax=axes[1, 0], cbar_kws={'label': '样本数'})
    axes[1, 0].set_xlabel('预测类别', fontsize=11)
    axes[1, 0].set_ylabel('真实类别', fontsize=11)
    axes[1, 0].set_title('混淆矩阵', fontsize=12, fontweight='bold')
    
    # 4. 每个类别的F1分数
    x_pos = range(len(CATEGORIES))
    bars = axes[1, 1].bar(x_pos, per_class_metrics['f1'], color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('类别', fontsize=11)
    axes[1, 1].set_ylabel('F1分数', fontsize=11)
    axes[1, 1].set_title('各类别F1分数', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(CATEGORIES, rotation=45, ha='right')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, per_class_metrics['f1']):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练结果图已保存: {save_path}")
    plt.close()

def main():
    # 检查数据集
    train_dir = os.path.join(DATA_DIR, 'train')
    if not os.path.exists(train_dir):
        print(f"\n❌ 错误: 数据集不存在！")
        print(f"   请确保数据集位于: {train_dir}")
        return
    
    print("\n🔧 加载预训练模型...")
    model_name = "google/vit-base-patch16-224"
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(CATEGORIES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    print("✅ 基础模型加载完成")
    
    # 配置LoRA
    print("\n🎯 配置LoRA...")
    lora_config = LoraConfig(
        r=16,  # LoRA秩
        lora_alpha=32,  # LoRA缩放因子
        target_modules=["query", "value"],  # 应用LoRA的模块
        lora_dropout=0.1,
        bias="none",
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✅ LoRA配置完成")
    
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
        learning_rate=2e-4,  # LoRA通常用更高的学习率
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(RESULTS_DIR, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
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
    print("\n🚀 开始LoRA微调...")
    print("=" * 70)
    
    train_result = trainer.train()
    
    # 保存模型
    print("\n💾 保存LoRA模型...")
    try:
        model.save_pretrained(MODEL_DIR)
        processor.save_pretrained(MODEL_DIR)
        print(f"✅ LoRA模型已保存到: {MODEL_DIR}")
        
        # 检查模型大小
        import glob
        model_files = glob.glob(os.path.join(MODEL_DIR, '*.bin')) + glob.glob(os.path.join(MODEL_DIR, '*.safetensors'))
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
    
    # 手动计算详细指标
    print("\n� 手动计算分类指标...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in trainer.get_eval_dataloader():
            pixel_values = batch['pixel_values'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import seaborn as sns
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # 每个类别的指标
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    print(f"   准确率: {accuracy:.4f}")
    print(f"   精确率: {precision:.4f}")
    print(f"   召回率: {recall:.4f}")
    print(f"   F1分数: {f1:.4f}")
    
    # 更新eval_metrics
    eval_metrics['eval_accuracy'] = accuracy
    eval_metrics['eval_precision'] = precision
    eval_metrics['eval_recall'] = recall
    eval_metrics['eval_f1'] = f1
    
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # 绘制训练结果（包含手动计算的指标）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(RESULTS_DIR, f'training_results_{timestamp}.png')
    
    # 准备绘图数据
    plot_data = {
        'log_history': trainer.state.log_history,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'per_class_metrics': {
            'precision': per_class_precision,
            'recall': per_class_recall,
            'f1': per_class_f1
        }
    }
    
    plot_training_results(plot_data, save_path=plot_path)
    
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
    print("✅ LoRA微调完成！")
    print("=" * 70)
    print(f"\n📁 模型保存位置: {MODEL_DIR}")
    print(f"📊 训练结果图: {plot_path}")
    print(f"📄 训练摘要: {summary_path}")
    print(f"\n🎯 最终准确率: {accuracy:.2%}")
    print(f"🎯 最终F1分数: {f1:.4f}")
    print(f"\n💡 LoRA优势: 模型大小从~330MB减少到~10MB！")

if __name__ == '__main__':
    main()
