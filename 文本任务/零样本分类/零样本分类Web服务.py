#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零样本分类 Web 服务 - 分类魔法师 🎯
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, send_file
from transformers import pipeline
import base64

BACKGROUND_PATH = r'背景.png'

print("=" * 70)
print("🎯 零样本分类 Web 服务 - 分类魔法师")
print("=" * 70)

print("\n🔮 正在加载分类模型...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("✅ 分类魔法师准备完毕！")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(CURRENT_DIR, '背景.png')

app = Flask(__name__)

background_base64 = ""
if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_base64 = base64.b64encode(f.read()).decode('utf-8')

HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 零样本分类 - 分类魔法师</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            background: url('data:image/png;base64,{background_base64}') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            padding: 20px;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        
        .falling-item {{
            position: fixed;
            font-size: 25px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.7;
        }}
        
        @keyframes fall {{
            0% {{
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.7;
            }}
            100% {{
                transform: translateY(100vh) rotate(360deg) scale(1.2);
                opacity: 0.3;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(156, 39, 176, 0.95) 0%, rgba(123, 31, 162, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(156, 39, 176, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 900px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(156, 39, 176, 0.6);
            position: relative;
            z-index: 10;
        }}
        
        h1 {{
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .subtitle {{
            text-align: center;
            color: #f3e5f5;
            margin-bottom: 30px;
            font-size: 1.2em;
        }}
        
        .input-area {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
        }}
        
        .input-group {{
            margin-bottom: 20px;
        }}
        
        .input-group label {{
            display: block;
            color: #7b1fa2;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        textarea, input[type="text"] {{
            width: 100%;
            padding: 15px;
            border: 2px solid #9c27b0;
            border-radius: 15px;
            font-size: 1em;
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            transition: all 0.3s;
        }}
        
        textarea {{
            resize: vertical;
            min-height: 100px;
        }}
        
        textarea:focus, input:focus {{
            outline: none;
            border-color: #7b1fa2;
            box-shadow: 0 0 15px rgba(156, 39, 176, 0.3);
        }}
        
        .hint {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        button {{
            width: 100%;
            padding: 18px;
            font-size: 1.3em;
            font-weight: bold;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 6px 20px rgba(156, 39, 176, 0.4);
            background: linear-gradient(135deg, #9c27b0 0%, #7b1fa2 100%);
            color: white;
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(156, 39, 176, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(243, 229, 245, 0.95) 0%, rgba(225, 190, 231, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #9c27b0;
        }}
        
        .category-item {{
            background: white;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #9c27b0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
        }}
        
        .category-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(156, 39, 176, 0.2);
        }}
        
        .category-name {{
            font-weight: bold;
            color: #7b1fa2;
            font-size: 1.1em;
        }}
        
        .category-score {{
            color: #9c27b0;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e1bee7;
            border-radius: 4px;
            margin-top: 8px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #9c27b0 0%, #7b1fa2 100%);
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 零样本分类</h1>
        <p class="subtitle">分类魔法师无需训练即可分类！</p>
        
        <div class="input-area">
            <div class="input-group">
                <label>📝 输入文本：</label>
                <textarea id="inputText" placeholder="请输入要分类的文本...&#10;例如：I love this movie, it's amazing!"></textarea>
            </div>
            
            <div class="input-group">
                <label>🏷️ 候选标签（用逗号分隔）：</label>
                <input type="text" id="labels" placeholder="例如：positive, negative, neutral" value="positive, negative, neutral">
                <div class="hint">💡 提示：可以输入任意标签，无需预先训练</div>
            </div>
            
            <button id="classifyBtn" onclick="classify()">
                🔮 开始分类
            </button>
        </div>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        const fallingItems = ['🎯', '🔮', '✨', '⭐', '🌟', '💫', '🎪', '🎭', '🎨', '🎬', '📊', '📈', '🏷️', '🔖', '📌'];
        
        function createFallingItem() {{
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 15 + 20) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => item.remove(), 7000);
        }}
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {{
            setTimeout(createFallingItem, i * 150);
        }}
        
        setInterval(createFallingItem, 150);
        
        async function classify() {{
            const inputText = document.getElementById('inputText').value.trim();
            const labelsText = document.getElementById('labels').value.trim();
            
            if (!inputText) {{
                alert('请输入要分类的文本！');
                return;
            }}
            
            if (!labelsText) {{
                alert('请输入候选标签！');
                return;
            }}
            
            const labels = labelsText.split(',').map(l => l.trim()).filter(l => l);
            
            if (labels.length === 0) {{
                alert('请至少输入一个标签！');
                return;
            }}
            
            const resultDiv = document.getElementById('result');
            const classifyBtn = document.getElementById('classifyBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #7b1fa2; font-size: 1.2em;">🔮 分类魔法师正在施法...</p>';
            resultDiv.style.display = 'block';
            classifyBtn.disabled = true;
            
            try {{
                const response = await fetch('/classify', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ text: inputText, labels: labels }})
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = '<p style="text-align: center; color: #d32f2f;">❌ ' + data.error + '</p>';
                }} else {{
                    displayResult(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = '<p style="text-align: center; color: #d32f2f;">❌ 分类失败: ' + error.message + '</p>';
            }} finally {{
                classifyBtn.disabled = false;
            }}
        }}
        
        function displayResult(data) {{
            let html = '<h3 style="color: #7b1fa2; margin-bottom: 20px; text-align: center;">✨ 分类结果</h3>';
            
            html += '<div style="background: white; padding: 15px; border-radius: 15px; margin-bottom: 20px;">';
            html += '<p style="color: #666; line-height: 1.6;">' + data.text + '</p>';
            html += '</div>';
            
            data.results.forEach((item, index) => {{
                const percentage = (item.score * 100).toFixed(1);
                html += `
                    <div class="category-item">
                        <div style="flex: 1;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="category-name">${{index + 1}}. ${{item.label}}</span>
                                <span class="category-score">${{percentage}}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${{percentage}}%"></div>
                            </div>
                        </div>
                    </div>
                `;
            }});
            
            document.getElementById('result').innerHTML = html;
        }}
    </script>
</body>
</html>
"""


@app.route('/static/background')
def background():
    if os.path.exists(BACKGROUND_PATH):
        return send_file(BACKGROUND_PATH, mimetype='image/png')
    else:
        return '', 404

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        text = data.get('text', '')
        labels = data.get('labels', [])
        
        if not text or not labels:
            return jsonify({'error': '请提供文本和标签'}), 400
        
        result = classifier(text, labels)
        
        results = []
        for label, score in zip(result['labels'], result['scores']):
            results.append({
                'label': label,
                'score': float(score)
            })
        
        return jsonify({
            'text': text,
            'results': results
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "=" * 70)
    print("🎯 启动分类魔法师...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:7002")
    print("🔮 分类魔法师在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:7002')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=7002, debug=False)
