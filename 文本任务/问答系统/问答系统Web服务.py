#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问答系统 Web 服务
提供网页界面和 API 接口
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import pipeline
import torch
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

app = Flask(__name__)

# 全局变量
qa_pipeline = None
model_info = {}

def load_model():
    """加载问答模型"""
    global qa_pipeline, model_info
    
    print("=" * 70)
    print("🤖 加载问答模型")
    print("=" * 70)
    
    # 按优先级尝试加载模型
    model_paths = [
        ("中文问答模型_BERT优化版", "BERT 优化版"),
        ("中文问答模型_高级版", "高级版"),
        ("中文问答模型", "简单版"),
        ("bert-base-chinese", "预训练 BERT")
    ]
    
    for model_path, model_desc in model_paths:
        if os.path.exists(model_path) or model_path == "bert-base-chinese":
            try:
                print(f"\n📥 尝试加载: {model_desc}")
                print(f"   路径: {model_path}")
                
                device = 0 if torch.cuda.is_available() else -1
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model_path,
                    device=device
                )
                
                model_info = {
                    "name": model_desc,
                    "path": model_path,
                    "device": "GPU" if device == 0 else "CPU",
                    "status": "已加载"
                }
                
                print(f"✅ 模型加载成功: {model_desc}")
                print(f"   设备: {model_info['device']}")
                return True
                
            except Exception as e:
                print(f"⚠️  加载失败: {str(e)[:100]}")
                continue
    
    print("❌ 所有模型加载失败")
    return False

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能问答系统</title>
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
            overflow-y: auto;
            overflow-x: hidden;
        }
        
        .falling-item {
            position: fixed;
            font-size: 24px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.7;
        }
        
        @keyframes fall {
            0% {
                transform: translateY(-10px) rotate(0deg);
                opacity: 0.7;
            }
            100% {
                transform: translateY(100vh) rotate(360deg);
                opacity: 0.2;
            }
        }
        
        .container {
            max-width: 900px;
            margin: 20px auto;
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
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .model-info {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 15px 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .model-info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .info-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .info-label {
            font-weight: 600;
            color: #667eea;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 1.1em;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        #context {
            min-height: 150px;
        }
        
        #question {
            min-height: 80px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            font-weight: 600;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        
        .result-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .answer-text {
            font-size: 1.5em;
            color: #333;
            margin-bottom: 15px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            font-weight: 500;
        }
        
        .confidence {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
        }
        
        .confidence-bar {
            flex: 1;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .confidence-label {
            font-weight: 600;
            color: #667eea;
            min-width: 80px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .examples {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .examples h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .example-item {
            padding: 10px;
            margin-bottom: 10px;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .example-item:hover {
            transform: translateX(5px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .example-label {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .error {
            background: #fee;
            border-left-color: #f44;
            color: #c33;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 智能问答系统</h1>
            <p>基于 Transformers 的中文问答模型</p>
        </div>
        
        <div class="model-info">
            <div class="model-info-grid">
                <div class="info-item">
                    <span class="info-label">模型:</span>
                    <span>{{ model_info.name }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">设备:</span>
                    <span>{{ model_info.device }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">状态:</span>
                    <span>{{ model_info.status }}</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <form id="qaForm">
                <div class="form-group">
                    <label for="context">📄 上下文（Context）</label>
                    <textarea id="context" name="context" placeholder="请输入包含答案的上下文内容..." required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="question">❓ 问题（Question）</label>
                    <textarea id="question" name="question" placeholder="请输入你的问题..." required></textarea>
                </div>
                
                <button type="submit" class="btn" id="submitBtn">
                    🔍 获取答案
                </button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>正在思考中...</p>
            </div>
            
            <div id="result"></div>
            
            <div class="examples">
                <h3>💡 示例问题</h3>
                <div class="example-item" onclick="loadExample(0)">
                    <div class="example-label">示例 1: 地理知识</div>
                    <div>北京是什么？</div>
                </div>
                <div class="example-item" onclick="loadExample(1)">
                    <div class="example-label">示例 2: 历史知识</div>
                    <div>谁修建了万里长城？</div>
                </div>
                <div class="example-item" onclick="loadExample(2)">
                    <div class="example-label">示例 3: 科技知识</div>
                    <div>深度学习在哪一年取得突破？</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const examples = [
            {
                context: "北京是中华人民共和国的首都，是全国的政治中心、文化中心。北京位于华北平原北部，背靠燕山，毗邻天津市和河北省。北京有着3000余年的建城史和850余年的建都史，是世界上拥有世界文化遗产数最多的城市。",
                question: "北京是什么？"
            },
            {
                context: "长城是中国古代的军事防御工程，是一道高大、坚固而连绵不断的长垣，用以限隔敌骑的行动。长城修筑的历史可上溯到西周时期。秦灭六国统一天下后，秦始皇连接和修缮战国长城，始有万里长城之称。",
                question: "谁修建了万里长城？"
            },
            {
                context: "深度学习是机器学习的一个分支，它基于人工神经网络的研究。2012年，深度学习在ImageNet图像识别竞赛中取得了巨大成功，错误率大幅降低，从此深度学习开始在学术界和工业界广泛应用。",
                question: "深度学习在哪一年取得突破？"
            }
        ];
        
        function loadExample(index) {
            const example = examples[index];
            document.getElementById('context').value = example.context;
            document.getElementById('question').value = example.question;
        }
        
        document.getElementById('qaForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const context = document.getElementById('context').value;
            const question = document.getElementById('question').value;
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            
            // 显示加载状态
            submitBtn.disabled = true;
            loading.style.display = 'block';
            resultDiv.innerHTML = '';
            
            try {
                const response = await fetch('/api/answer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ context, question })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const confidence = (data.score * 100).toFixed(2);
                    const confidenceColor = confidence > 50 ? '#667eea' : confidence > 20 ? '#f39c12' : '#e74c3c';
                    
                    resultDiv.innerHTML = `
                        <div class="result">
                            <div class="result-title">✨ 答案</div>
                            <div class="answer-text">${data.answer}</div>
                            <div class="confidence">
                                <span class="confidence-label">置信度:</span>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${confidence}%; background: ${confidenceColor}">
                                        ${confidence}%
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <div class="result-title">❌ 错误</div>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="result error">
                        <div class="result-title">❌ 请求失败</div>
                        <p>${error.message}</p>
                    </div>
                `;
            } finally {
                submitBtn.disabled = false;
                loading.style.display = 'none';
            }
        });
        
        // 飘落动画
        const emojis = ['🤖', '💡', '📚', '✨', '🎯', '💬', '🔍', '📝', '🌟', '💭'];
        
        function createFallingItem() {
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = emojis[Math.floor(Math.random() * emojis.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 10 + 20) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => {
                item.remove();
            }, 7000);
        }
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {
            setTimeout(createFallingItem, i * 150);
        }
        
        // 持续创建新元素
        setInterval(createFallingItem, 150);
    </script>
</body>
</html>
"""

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

@app.route('/api/answer', methods=['POST'])
def answer():
    """问答 API"""
    try:
        data = request.json
        context = data.get('context', '').strip()
        question = data.get('question', '').strip()
        
        if not context or not question:
            return jsonify({
                'success': False,
                'error': '上下文和问题不能为空'
            })
        
        # 调用模型
        start_time = time.time()
        result = qa_pipeline(question=question, context=context)
        elapsed_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'score': float(result['score']),
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

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 启动问答系统 Web 服务")
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
    print("   2. 输入上下文和问题")
    print("   3. 点击「获取答案」按钮")
    print("   4. 或点击示例问题快速测试")
    print("\n⚠️  按 Ctrl+C 停止服务")
    print("=" * 70 + "\n")
    
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=False)
