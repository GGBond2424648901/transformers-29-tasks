#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5 智能客服 Web 服务
提供网页聊天界面
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, render_template_string, send_file
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

app = Flask(__name__)

# 全局变量
model = None
tokenizer = None
model_info = {}

# ============================================================================
# 加载模型
# ============================================================================

def load_model():
    """加载 Qwen2.5 + LoRA 模型"""
    global model, tokenizer, model_info
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("🤖 加载 Qwen2.5 客服模型")
    print("=" * 70)
    
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_path = os.path.join(script_dir, "output/chatglm-customer-lora")
    
    try:
        # 加载 tokenizer
        print("\n📥 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        print("✅ Tokenizer 加载成功")
        
        # 加载基础模型
        print("\n📥 加载基础模型...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✅ 基础模型加载成功")
        
        # 加载 LoRA 权重
        if os.path.exists(lora_path):
            print(f"\n📥 加载 LoRA 权重: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            model_info = {
                "name": "ChatGLM-6B 客服版（LoRA 微调）",
                "type": "微调模型",
                "status": "已加载"
            }
            print("✅ LoRA 权重加载成功")
        else:
            model_info = {
                "name": "ChatGLM-6B（原始）",
                "type": "基础模型",
                "status": "已加载（未微调）"
            }
            print("⚠️  未找到 LoRA 权重，使用基础模型")
        
        model = model.eval()
        
        device = "GPU" if torch.cuda.is_available() else "CPU"
        model_info["device"] = device
        
        print(f"\n✅ 模型加载完成")
        print(f"   设备: {device}")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("\n💡 提示：")
        print("1. 首次运行会自动下载 ChatGLM-6B（约 12GB）")
        print("2. 请确保网络连接正常")
        print("3. 或先运行训练脚本生成 LoRA 权重")
        return False

# ============================================================================
# HTML 模板
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能客服系统 - ChatGLM</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-image: url('/static/background');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .model-info {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 15px 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .chat-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            height: 600px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.assistant .message-content {
            background: #f0f0f0;
            color: #333;
        }
        
        .chat-input-container {
            padding: 20px;
            border-top: 1px solid #e0e0e0;
        }
        
        .chat-input-wrapper {
            display: flex;
            gap: 10px;
        }
        
        #userInput {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #userInput:focus {
            border-color: #667eea;
        }
        
        #sendBtn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s;
        }
        
        #sendBtn:hover {
            transform: translateY(-2px);
        }
        
        #sendBtn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 10px;
            color: #667eea;
        }
        
        .quick-questions {
            padding: 15px 20px;
            border-top: 1px solid #e0e0e0;
        }
        
        .quick-questions h4 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .quick-btn {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background: #f0f0f0;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        
        .quick-btn:hover {
            background: #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 智能客服系统</h1>
            <p>基于 ChatGLM-6B LoRA 微调</p>
        </div>
        
        <div class="model-info">
            <strong>模型:</strong> {{ model_info.name }} | 
            <strong>类型:</strong> {{ model_info.type }} | 
            <strong>设备:</strong> {{ model_info.device }} | 
            <strong>状态:</strong> {{ model_info.status }}
        </div>
        
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    <div class="message-content">
                        您好！我是智能客服助手，有什么可以帮您？
                    </div>
                </div>
            </div>
            
            <div class="loading" id="loading">
                正在思考中...
            </div>
            
            <div class="quick-questions">
                <h4>💡 快速提问</h4>
                <button class="quick-btn" onclick="sendQuickQuestion('如何退货？')">如何退货？</button>
                <button class="quick-btn" onclick="sendQuickQuestion('发货需要多久？')">发货需要多久？</button>
                <button class="quick-btn" onclick="sendQuickQuestion('支持哪些支付方式？')">支持哪些支付方式？</button>
                <button class="quick-btn" onclick="sendQuickQuestion('如何联系客服？')">如何联系客服？</button>
            </div>
            
            <div class="chat-input-container">
                <div class="chat-input-wrapper">
                    <input type="text" id="userInput" placeholder="输入您的问题..." onkeypress="handleKeyPress(event)">
                    <button id="sendBtn" onclick="sendMessage()">发送</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function addMessage(content, isUser) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const sendBtn = document.getElementById('sendBtn');
            const loading = document.getElementById('loading');
            const question = input.value.trim();
            
            if (!question) return;
            
            // 显示用户消息
            addMessage(question, true);
            input.value = '';
            
            // 禁用输入
            sendBtn.disabled = true;
            input.disabled = true;
            loading.style.display = 'block';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addMessage(data.response, false);
                } else {
                    addMessage('抱歉，出现了错误：' + data.error, false);
                }
            } catch (error) {
                addMessage('抱歉，网络错误：' + error.message, false);
            } finally {
                sendBtn.disabled = false;
                input.disabled = false;
                loading.style.display = 'none';
                input.focus();
            }
        }
        
        function sendQuickQuestion(question) {
            document.getElementById('userInput').value = question;
            sendMessage();
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
"""

# ============================================================================
# 路由
# ============================================================================

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE, model_info=model_info)

@app.route('/static/background')
def background():
    """提供背景图片"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bg_path = os.path.join(script_dir, '背景.png')
    return send_file(bg_path, mimetype='image/png')

@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天 API"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': '问题不能为空'
            })
        
        # 构建 Qwen2.5 格式的提示
        messages = [
            {"role": "system", "content": "你是一个专业的智能客服助手，请根据用户的问题提供准确、友好的回答。"},
            {"role": "user", "content": question}
        ]
        
        # 使用 tokenizer 的 apply_chat_template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 调用模型生成
        start_time = time.time()
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                do_sample=True
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        elapsed_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'response': response.strip(),
            'time': f"{elapsed_time:.2f}s"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/info', methods=['GET'])
def info():
    """获取模型信息"""
    return jsonify(model_info)

# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 启动 Qwen2.5 智能客服系统")
    print("=" * 70)
    
    # 加载模型
    if not load_model():
        print("\n❌ 模型加载失败，无法启动服务")
        exit(1)
    
    print("\n" + "=" * 70)
    print("✅ 服务启动成功！")
    print("=" * 70)
    print("\n📱 访问地址:")
    print("   本地: http://127.0.0.1:5000")
    print("   局域网: http://0.0.0.0:5000")
    print("\n💡 使用说明:")
    print("   1. 在浏览器中打开上述地址")
    print("   2. 输入问题进行对话")
    print("   3. 或点击快速提问按钮")
    print("\n⚠️  按 Ctrl+C 停止服务")
    print("=" * 70 + "\n")
    
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=False)
