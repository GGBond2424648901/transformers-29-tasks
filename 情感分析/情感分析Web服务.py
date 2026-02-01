#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分析 Web 服务
提供网页界面和 API 接口
"""

import os
from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import torch.nn.functional as F
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

app = Flask(__name__)

# 全局变量
model = None
tokenizer = None
classifier = None
model_info = {}

# ============================================================================
# 加载模型
# ============================================================================

def load_model():
    """加载情感分析模型"""
    global model, tokenizer, classifier, model_info
    
    print("=" * 70)
    print("🤖 加载情感分析模型")
    print("=" * 70)
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "my_sentiment_model")
    
    try:
        print(f"\n📥 模型路径: {model_path}")
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            print(f"❌ 错误：找不到模型文件夹")
            print(f"   期望位置: {model_path}")
            print(f"\n💡 请先训练模型：双击 训练模型.bat")
            return False
        
        # 加载模型
        print("\n📥 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer 加载成功")
        
        print("\n📥 加载模型...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ 模型加载成功")
        
        # 创建 pipeline
        classifier = pipeline("text-classification", model=model_path)
        
        # 设置设备
        device = "GPU" if torch.cuda.is_available() else "CPU"
        model_info["device"] = device
        model_info["model_path"] = model_path
        
        print(f"\n✅ 模型加载完成")
        print(f"   设备: {device}")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
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
    <title>情感分析系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
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
        
        .model-info {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 15px 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .model-info-content {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .info-item {
            text-align: center;
        }
        
        .info-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .info-value {
            font-size: 1.1em;
            font-weight: bold;
            color: #667eea;
        }
        
        .main-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        label {
            display: block;
            margin-bottom: 10px;
            color: #333;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
        }
        
        button {
            flex: 1;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .clear-btn {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            display: none;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result.positive {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-left: 5px solid #4caf50;
        }
        
        .result.negative {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-left: 5px solid #f44336;
        }
        
        .result-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .result-icon {
            font-size: 2.5em;
            margin-right: 15px;
        }
        
        .result-title {
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .result-content {
            margin-top: 15px;
        }
        
        .result-text {
            background: rgba(255, 255, 255, 0.7);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-style: italic;
        }
        
        .result-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .detail-item {
            background: rgba(255, 255, 255, 0.7);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .detail-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .detail-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-weight: bold;
        }
        
        .loading-spinner {
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
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
        }
        
        .examples h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .example-btn {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
        }
        
        .example-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 情感分析系统</h1>
            <p>基于深度学习的中文情感分析</p>
        </div>
        
        <div class="model-info">
            <div class="model-info-content">
                <div class="info-item">
                    <div class="info-label">运行设备</div>
                    <div class="info-value">{{ model_info.device }}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">模型状态</div>
                    <div class="info-value">✅ 已加载</div>
                </div>
            </div>
        </div>
        
        <div class="main-card">
            <div class="input-section">
                <label for="textInput">请输入要分析的文本：</label>
                <textarea id="textInput" placeholder="例如：这个产品质量很好，我很满意！"></textarea>
            </div>
            
            <div class="button-group">
                <button id="analyzeBtn" onclick="analyze()">🔍 开始分析</button>
                <button class="clear-btn" onclick="clearAll()">🗑️ 清空</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <div>正在分析中...</div>
            </div>
            
            <div class="result" id="result"></div>
            
            <div class="examples">
                <h3>💡 示例文本（点击快速填充）</h3>
                <span class="example-btn" onclick="fillExample('这个产品质量很好，我很满意！')">正面示例1</span>
                <span class="example-btn" onclick="fillExample('服务态度非常好，值得推荐')">正面示例2</span>
                <span class="example-btn" onclick="fillExample('物流很快，包装完好')">正面示例3</span>
                <span class="example-btn" onclick="fillExample('质量太差了，非常失望')">负面示例1</span>
                <span class="example-btn" onclick="fillExample('客服态度恶劣，不推荐')">负面示例2</span>
                <span class="example-btn" onclick="fillExample('价格贵，性价比低')">负面示例3</span>
            </div>
        </div>
    </div>
    
    <script>
        function fillExample(text) {
            document.getElementById('textInput').value = text;
        }
        
        function clearAll() {
            document.getElementById('textInput').value = '';
            document.getElementById('result').style.display = 'none';
        }
        
        async function analyze() {
            const text = document.getElementById('textInput').value.trim();
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            if (!text) {
                alert('请输入要分析的文本！');
                return;
            }
            
            // 显示加载状态
            analyzeBtn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResult(data);
                } else {
                    alert('分析失败：' + data.error);
                }
            } catch (error) {
                alert('请求失败：' + error.message);
            } finally {
                analyzeBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        function displayResult(data) {
            const result = document.getElementById('result');
            const sentiment = data.sentiment;
            const isPositive = sentiment === '正面';
            
            result.className = 'result ' + (isPositive ? 'positive' : 'negative');
            result.innerHTML = `
                <div class="result-header">
                    <div class="result-icon">${isPositive ? '😊' : '😞'}</div>
                    <div class="result-title">情感倾向：${sentiment}</div>
                </div>
                <div class="result-content">
                    <div class="result-text">"${data.text}"</div>
                    <div class="result-details">
                        <div class="detail-item">
                            <div class="detail-label">置信度</div>
                            <div class="detail-value">${(data.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">正面概率</div>
                            <div class="detail-value">${(data.probabilities.positive * 100).toFixed(1)}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">负面概率</div>
                            <div class="detail-value">${(data.probabilities.negative * 100).toFixed(1)}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">分析时间</div>
                            <div class="detail-value">${data.time}</div>
                        </div>
                    </div>
                </div>
            `;
            result.style.display = 'block';
        }
        
        // 支持回车键提交
        document.getElementById('textInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                analyze();
            }
        });
        
        // 飘落动画
        const emojis = ['😊', '😢', '😡', '😍', '😱', '🎭', '💖', '💔', '✨', '🌟'];
        
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

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """情感分析 API"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': '文本不能为空'
            })
        
        # 开始计时
        start_time = time.time()
        
        # 使用模型进行预测
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        negative_prob = probs[0][0].item()
        positive_prob = probs[0][1].item()
        
        elapsed_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'text': text,
            'sentiment': '正面' if positive_prob > negative_prob else '负面',
            'confidence': max(positive_prob, negative_prob),
            'probabilities': {
                'negative': negative_prob,
                'positive': positive_prob
            },
            'time': f"{elapsed_time:.3f}s"
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
    print("🚀 启动情感分析系统")
    print("=" * 70)
    
    # 加载模型
    if not load_model():
        print("\n❌ 模型加载失败，无法启动服务")
        exit(1)
    
    print("\n" + "=" * 70)
    print("✅ 服务启动成功！")
    print("=" * 70)
    print("\n📱 访问地址: http://127.0.0.1:5000")
    print("💡 提示: 按 Ctrl+C 停止服务\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
