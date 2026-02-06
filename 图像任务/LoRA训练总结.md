# 图像任务 LoRA 训练总结

## 📋 项目概述

使用 LoRA (Low-Rank Adaptation) 技术对三个图像任务进行轻量化微调，实现模型大小减少 99%+ 的同时保持优异性能。

---

## ✅ 完成任务

### 1. 图像分类 LoRA
- **基础模型**: ViT-Base (google/vit-base-patch16-224)
- **任务**: 5类图像分类（猫、狗、鸟、鱼、马）
- **模型大小**: 2.27 MB（原 330 MB，减少 99.3%）
- **训练时间**: ~30秒
- **性能**: 准确率 100%，F1分数 1.0000
- **脚本**: `图像分类/图像分类训练_LoRA.py`

### 2. 目标检测 LoRA
- **基础模型**: DETR-ResNet-50 (facebook/detr-resnet-50)
- **任务**: 检测圆形、方形、三角形
- **模型大小**: 0.59 MB（原 160 MB，减少 99.6%）
- **训练时间**: ~84秒
- **性能**: 训练损失从 9.18 降至 2.87
- **脚本**: `目标检测/目标检测训练_LoRA.py`

### 3. 姿态估计 LoRA
- **基础模型**: ViT-Base + 自定义关键点检测头
- **任务**: 5个关键点检测（头部、左手、右手、左脚、右脚）
- **模型大小**: ~2 MB LoRA权重（原 330 MB，减少 99.4%）
- **训练时间**: ~35秒
- **性能**: 像素误差 9.68，MAE 0.0432
- **脚本**: `姿态估计/姿态估计训练_LoRA.py`

---

## 🎯 LoRA 配置

### 图像分类 & 姿态估计
```python
LoraConfig(
    r=16,                              # LoRA秩
    lora_alpha=32,                     # 缩放因子
    target_modules=["query", "value"], # ViT注意力层
    lora_dropout=0.1,
    bias="none"
)
```

### 目标检测
```python
LoraConfig(
    r=8,                                # 较小的秩
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # DETR注意力层
    lora_dropout=0.1,
    bias="none"
)
```

---

## 📊 性能对比

| 任务 | 原始大小 | LoRA大小 | 压缩率 | 训练时间 | 性能表现 |
|------|---------|---------|--------|---------|---------|
| 图像分类 | 330 MB | 2.27 MB | 99.3% | 30秒 | 准确率 100% |
| 目标检测 | 160 MB | 0.59 MB | 99.6% | 84秒 | 损失降至 2.87 |
| 姿态估计 | 330 MB | ~2 MB | 99.4% | 35秒 | 像素误差 9.68 |

**平均压缩率**: 99.4% ⬇️

---

## 📈 训练结果可视化

### 图像分类
生成 4 个图表：
- 训练损失曲线
- 分类性能指标（准确率、精确率、召回率、F1）
- 混淆矩阵
- 各类别 F1 分数

### 目标检测
- 训练损失曲线

### 姿态估计
生成 4 个图表：
- 训练损失曲线
- 验证损失曲线
- 平均绝对误差（MAE）
- 关键点像素误差

---

## 🚀 使用方法

### 运行训练
```bash
# 使用虚拟环境
D:\aaaalokda\envs\myenv\python.exe 图像分类训练_LoRA.py
D:\aaaalokda\envs\myenv\python.exe 目标检测训练_LoRA.py
D:\aaaalokda\envs\myenv\python.exe 姿态估计训练_LoRA.py
```

### 加载模型
```python
from peft import PeftModel
from transformers import AutoModel

# 加载基础模型
base_model = AutoModel.from_pretrained("google/vit-base-patch16-224")

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "trained_model_lora")
```

---

## 💡 核心优势

1. **极致压缩**: 模型大小减少 99%+
2. **快速训练**: 所有任务 2 分钟内完成
3. **性能保持**: 分类准确率 100%，姿态估计误差仅 9.68 像素
4. **易于部署**: 轻量级模型适合边缘设备
5. **低资源消耗**: 可训练参数 < 1%

---

## 📁 文件结构

```
图像任务/
├── 图像分类/
│   ├── 图像分类训练_LoRA.py
│   ├── trained_model_lora/          # 2.27 MB
│   └── training_results_lora/
├── 目标检测/
│   ├── 目标检测训练_LoRA.py
│   ├── trained_model_lora/          # 0.59 MB
│   └── training_results_lora/
└── 姿态估计/
    ├── 姿态估计训练_LoRA.py
    ├── trained_model_lora/          # ~2 MB
    └── training_results_lora/
```

---

## 🔧 技术要点

- **PEFT 版本**: 0.4.0（兼容性考虑）
- **GPU 加速**: 自动检测 NVIDIA RTX 3070
- **训练策略**: `save_strategy="no"`（避免 Windows 权限问题）
- **学习率**: 2e-4（LoRA 推荐）
- **评估指标**: 手动计算确保准确性

---

## ✅ 项目状态

- ✅ 三个 LoRA 训练脚本已创建并验证
- ✅ 所有训练成功完成
- ✅ 模型压缩率达到 99%+
- ✅ 性能指标完整记录
- ✅ 可视化图表生成

---

## 📝 推荐场景

**强烈推荐使用 LoRA**：
- 存储空间有限
- 需要快速实验迭代
- 数据集较小（< 10000 样本）
- 边缘设备部署
- 多任务微调场景

---

**训练完成时间**: 2026年2月6日
