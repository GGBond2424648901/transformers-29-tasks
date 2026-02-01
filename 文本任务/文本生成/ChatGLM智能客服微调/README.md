# 🤖 ChatGLM-6B 智能客服 LoRA 微调项目

## 📋 项目简介

使用 LoRA（Low-Rank Adaptation）技术微调 ChatGLM-6B 模型，打造专属的智能客服助手。

### 🎯 项目特点

- ✅ **低资源需求**：LoRA 微调只需 8-12GB 显存
- ✅ **快速训练**：相比全量微调快 3-5 倍
- ✅ **效果优秀**：接近全量微调的效果
- ✅ **易于部署**：只需保存 LoRA 权重（几十 MB）

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 进入项目目录
cd 实战训练/文本任务/文本生成/ChatGLM智能客服微调

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据（已提供示例）

查看 `data/train.json` 和 `data/dev.json`

### 3. 开始训练

```bash
# 方法 1：双击批处理文件
开始训练.bat

# 方法 2：命令行
D:\aaaalokda\envs\myenv\python.exe chatglm_lora_finetune.py
```

### 4. 测试模型

```bash
# 测试微调后的模型
D:\aaaalokda\envs\myenv\python.exe test_model.py
```

### 5. 启动 Web 服务

```bash
# 启动客服 Web 界面
启动客服系统.bat
```

---

## 📁 项目结构

```
ChatGLM智能客服微调/
├── README.md                      # 本文件
├── requirements.txt               # 依赖包
├── chatglm_lora_finetune.py      # LoRA 微调脚本
├── test_model.py                  # 模型测试脚本
├── customer_service_web.py        # Web 服务
├── 开始训练.bat                   # 一键训练
├── 启动客服系统.bat               # 一键启动 Web
│
├── data/                          # 训练数据
│   ├── train.json                # 训练集（100条客服对话）
│   ├── dev.json                  # 验证集（20条）
│   └── 数据说明.md               # 数据格式说明
│
├── output/                        # 训练输出
│   └── chatglm-customer-lora/    # LoRA 权重
│
└── docs/                          # 文档
    ├── 训练指南.md
    ├── 数据准备指南.md
    └── 部署指南.md
```

---

## 💾 数据格式

### 训练数据格式（JSON）

```json
[
  {
    "instruction": "用户问：如何退货？",
    "input": "",
    "output": "您好！退货流程如下：\n1. 登录您的账户\n2. 进入"我的订单"页面\n3. 选择需要退货的商品\n4. 点击"申请退货"\n5. 填写退货原因\n6. 等待审核（1-2个工作日）\n\n如有疑问，请联系在线客服或拨打 400-xxx-xxxx"
  },
  {
    "instruction": "用户问：发货需要多久？",
    "input": "",
    "output": "您好！发货时间如下：\n- 现货商品：下单后24小时内发货\n- 预售商品：按预售页面显示时间发货\n- 定制商品：7-15个工作日\n\n您可以在订单详情中查看具体发货时间。"
  }
]
```

---

## ⚙️ 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | ChatGLM-6B | 6B 参数的对话模型 |
| LoRA Rank | 8 | 低秩矩阵的秩 |
| LoRA Alpha | 32 | 缩放因子 |
| 学习率 | 5e-4 | 较高的学习率 |
| Batch Size | 4 | 每批次样本数 |
| Epochs | 3 | 训练轮数 |
| Max Length | 512 | 最大序列长度 |

---

## 📊 资源需求

### 显存需求

| 配置 | 显存 | 训练时间（100条数据） |
|------|------|---------------------|
| LoRA (推荐) | 8-12GB | 10-15分钟 |
| 全量微调 | 40GB+ | 60-90分钟 |

### 你的配置

- GPU: NVIDIA GeForce RTX 3070 Laptop (8GB)
- ✅ 完全满足 LoRA 微调需求

---

## 🎯 应用场景

### 1. 电商客服
- 订单查询
- 退换货咨询
- 物流追踪
- 售后服务

### 2. 技术支持
- 产品使用指导
- 故障排查
- 功能咨询
- 升级说明

### 3. 金融客服
- 账户查询
- 业务办理
- 政策咨询
- 风险提示

---

## 📈 训练流程

```
1. 数据准备
   ↓
2. 加载 ChatGLM-6B 基础模型
   ↓
3. 添加 LoRA 适配器
   ↓
4. 训练 LoRA 权重（只训练少量参数）
   ↓
5. 保存 LoRA 权重
   ↓
6. 测试效果
   ↓
7. 部署使用
```

---

## 🔧 自定义训练

### 修改训练数据

编辑 `data/train.json`，添加你的客服对话：

```json
{
  "instruction": "用户问：你的问题",
  "input": "",
  "output": "客服回答：你的回答"
}
```

### 调整训练参数

编辑 `chatglm_lora_finetune.py`：

```python
# 修改这些参数
training_args = TrainingArguments(
    num_train_epochs=3,        # 训练轮数
    per_device_train_batch_size=4,  # 批次大小
    learning_rate=5e-4,        # 学习率
    ...
)
```

---

## 🌐 Web 服务

### 功能特点

- 🎨 美观的聊天界面
- 💬 实时对话
- 📝 对话历史
- 🔄 多轮对话支持
- 📊 响应时间显示

### 访问地址

启动后访问：`http://127.0.0.1:5000`

---

## 💡 使用技巧

### 1. 数据质量最重要

- ✅ 提供真实的客服对话
- ✅ 回答要专业、准确
- ✅ 保持统一的风格
- ❌ 避免错误或矛盾的信息

### 2. 训练数据量

| 数据量 | 效果 | 推荐场景 |
|--------|------|---------|
| 50-100条 | 基础 | 快速测试 |
| 200-500条 | 良好 | 小型客服 |
| 1000+条 | 优秀 | 生产环境 |

### 3. 持续优化

- 收集用户反馈
- 补充新的对话数据
- 定期重新训练

---

## 🐛 常见问题

### Q1: 显存不足怎么办？

A: 降低 batch size 或 max_length：
```python
per_device_train_batch_size=2  # 从 4 改为 2
max_length=256  # 从 512 改为 256
```

### Q2: 训练很慢？

A: 检查是否使用 GPU：
```python
import torch
print(torch.cuda.is_available())  # 应该输出 True
```

### Q3: 模型回答不准确？

A: 可能原因：
1. 训练数据太少（增加到 200+ 条）
2. 训练轮数不够（增加到 5-10 轮）
3. 数据质量不高（检查训练数据）

### Q4: 如何添加新的客服知识？

A: 在 `data/train.json` 中添加新的对话对，然后重新训练。

---

## 📚 扩展阅读

- [ChatGLM-6B 官方文档](https://github.com/THUDM/ChatGLM-6B)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 库文档](https://huggingface.co/docs/peft)

---

## 🎉 快速命令

```bash
# 安装依赖
pip install -r requirements.txt

# 训练模型
开始训练.bat

# 测试模型
D:\aaaalokda\envs\myenv\python.exe test_model.py

# 启动 Web 服务
启动客服系统.bat
```

---

**准备好了吗？双击 `开始训练.bat` 开始训练你的智能客服！** 🚀
