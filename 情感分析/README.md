# 情感分析实战项目

基于 Transformers 的中文情感分析完整实战项目。

## 📁 项目结构

```
情感分析/
├── README.md                    # 本文档
├── Trainer_API_详解.md          # Trainer API 详细说明
├── 训练模型详解.md              # 模型结构和原理详解
├── 模型使用指南.md              # 完整使用指南
│
├── Trainer_实战示例.py          # 训练脚本
├── 使用训练好的模型.py          # 命令行预测脚本
├── 模型部署示例.py              # Flask API 服务
├── 快速测试API.py               # API 测试脚本
├── 测试API客户端.py             # 完整 API 测试
├── 测试API网页.html             # 网页版测试界面
│
├── 启动API服务.bat              # 一键启动 API
├── 测试API.bat                  # 一键测试 API
│
└── my_sentiment_model/          # 训练好的模型
    ├── config.json              # 模型配置
    ├── model.safetensors        # 模型权重
    ├── tokenizer.json           # 分词器
    └── tokenizer_config.json    # 分词器配置
```

---

## 🚀 快速开始

### ⚠️ 重要提示

**所有脚本必须在 `实战训练/情感分析/` 目录下运行！**

```bash
# 进入项目目录
cd D:\transformers训练\transformers-main\实战训练\情感分析

# 然后运行脚本
D:\aaaalokda\envs\myenv\python.exe Trainer_实战示例.py
```

或者直接双击批处理文件（.bat），它们会自动切换到正确的目录。

---

### 环境要求

- Python 3.9+
- PyTorch 2.3.1+cu121
- Transformers 5.0.0
- Flask 3.1.2
- flask-cors 6.0.2

### 安装依赖

```bash
# 使用项目环境
D:\aaaalokda\envs\myenv\python.exe -m pip install transformers torch datasets accelerate flask flask-cors
```

---

## 📖 使用方法

### 1️⃣ 训练模型

**方法 A: 双击批处理文件（推荐）**
```
双击：训练模型.bat
```

**方法 B: 命令行运行**
```bash
# 先进入项目目录
cd D:\transformers训练\transformers-main\实战训练\情感分析

# 运行训练脚本
D:\aaaalokda\envs\myenv\python.exe Trainer_实战示例.py
```

**说明**：
- 使用 BERT-base-chinese 预训练模型
- 在情感分类数据集上微调
- 训练结果保存在 `my_sentiment_model/`

---

### 2️⃣ 使用模型预测

#### 方法 A: 双击批处理文件（推荐）

```
双击：使用模型预测.bat
```

#### 方法 B: 命令行预测

```bash
# 先进入项目目录
cd D:\transformers训练\transformers-main\实战训练\情感分析

# 运行预测脚本
D:\aaaalokda\envs\myenv\python.exe 使用训练好的模型.py
```

**功能**：
- 批量预测预设文本
- 交互式输入预测

#### 方法 B: Python 代码

```python
from transformers import pipeline

# 注意：需要在 实战训练/情感分析/ 目录下运行
# 使用相对路径（不带 ./）
classifier = pipeline("text-classification", model="my_sentiment_model")

# 预测
result = classifier("这个产品很好")
print(result)  # [{'label': 'LABEL_1', 'score': 0.9468}]
```

---

### 3️⃣ 部署 API 服务

#### 启动服务

**方法 1: 双击批处理文件（推荐）**
```
双击：启动API服务.bat
```

**方法 2: 命令行运行**
```bash
# 先进入项目目录
cd D:\transformers训练\transformers-main\实战训练\情感分析

# 运行 API 服务
D:\aaaalokda\envs\myenv\python.exe 模型部署示例.py
```

**服务地址**：
- 本地：http://localhost:5000
- 局域网：http://10.146.20.177:5000

#### 测试 API

**方法 1: 网页测试（推荐）**
- 双击打开 `测试API网页.html`
- 在浏览器中输入文本测试

**方法 2: Python 脚本测试**
```bash
# 双击批处理文件
测试API.bat

# 或命令行运行
D:\aaaalokda\envs\myenv\python.exe 快速测试API.py
```

**方法 3: curl 测试**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"这个产品很好\"}"
```

---

## 📊 API 接口

### 1. 健康检查

```
GET /health
```

**响应**：
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### 2. 单个文本预测

```
POST /predict
Content-Type: application/json

{
  "text": "这个产品很好",
  "use_pipeline": false
}
```

**响应**：
```json
{
  "text": "这个产品很好",
  "sentiment": "正面",
  "confidence": 0.9468,
  "probabilities": {
    "negative": 0.0532,
    "positive": 0.9468
  }
}
```

---

### 3. 批量文本预测

```
POST /batch_predict
Content-Type: application/json

{
  "texts": ["质量很好", "太差了", "一般般"]
}
```

**响应**：
```json
{
  "count": 3,
  "results": [
    {
      "text": "质量很好",
      "label": "LABEL_1",
      "score": 0.9468,
      "sentiment": "正面"
    },
    ...
  ]
}
```

---

## 📚 文档说明

### Trainer_API_详解.md
- Trainer API 的概念和优势
- 与传统训练方式的对比
- 详细参数说明

### 训练模型详解.md
- 模型文件结构详解
- BERT 架构原理
- 参数量计算
- 模型可视化

### 模型使用指南.md
- 完整使用教程
- 各种应用场景
- 性能优化技巧
- 故障排查

---

## 🎯 模型信息

### 基础模型
- **名称**: bert-base-chinese
- **参数量**: 102M
- **隐藏层**: 12 层 Transformer
- **词汇量**: 21,128 个中文词汇

### 训练配置
- **任务**: 二分类（正面/负面）
- **优化器**: AdamW
- **学习率**: 2e-5
- **批量大小**: 8
- **训练轮数**: 3

### 模型性能
- **测试准确率**: ~75%（演示数据）
- **推理速度**: ~50ms/样本（CPU）
- **模型大小**: ~414 MB

---

## 💡 使用技巧

### 1. GPU 加速

```python
import torch

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移到 GPU
model.to(device)
```

### 2. 批量处理

```python
# 使用 pipeline 批处理
classifier = pipeline("text-classification", model="./my_sentiment_model", batch_size=32)
results = classifier(texts)
```

### 3. 模型量化

```python
# 动态量化（减小 4 倍）
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## ⚠️ 注意事项

1. **模型限制**
   - 最大长度：512 个 token（约 256 个中文字符）
   - 仅支持中文文本
   - 仅支持二分类

2. **性能优化**
   - GPU 比 CPU 快 20-50 倍
   - 批量处理比单个快 5-10 倍
   - 使用 FP16 可以再快 2 倍

3. **准确率提升**
   - 使用更多真实数据训练
   - 增加训练轮数
   - 调整学习率和批量大小

---

## 🔧 故障排查

### 问题 1: 模型加载失败

```
FileNotFoundError: config.json not found
```

**解决**：确保 `my_sentiment_model/` 文件夹完整

---

### 问题 2: API 连接失败

```
ConnectionError: Failed to establish connection
```

**解决**：
1. 确认 API 服务已启动
2. 检查端口 5000 是否被占用
3. 尝试使用 127.0.0.1 而不是 localhost

---

### 问题 3: CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决**：
- 减小批量大小
- 使用 CPU 模式
- 使用模型量化

---

## 📈 下一步

### 提升模型性能
1. 使用更多真实数据训练
2. 尝试其他预训练模型（RoBERTa、ELECTRA）
3. 调整超参数

### 扩展功能
1. 支持多分类（正面、中性、负面）
2. 添加情感强度评分
3. 支持多语言

### 生产部署
1. 使用 Gunicorn/uWSGI 替代 Flask 开发服务器
2. 添加负载均衡
3. 使用 Docker 容器化
4. 部署到云服务（AWS、阿里云）

---

## 📞 技术支持

- **Transformers 文档**: https://huggingface.co/docs/transformers
- **BERT 论文**: https://arxiv.org/abs/1810.04805
- **Hugging Face Hub**: https://huggingface.co/models

---

## 📝 更新日志

### v1.0.0 (2026-01-31)
- ✅ 完成基础训练脚本
- ✅ 实现命令行预测
- ✅ 部署 Flask API 服务
- ✅ 创建网页测试界面
- ✅ 编写完整文档

---

**祝你使用愉快！** 🎉
