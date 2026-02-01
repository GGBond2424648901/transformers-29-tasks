#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本摘要 Web 服务 - 摘要精灵 📄
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
import base64

BACKGROUND_PATH = r'背景.png'

print("=" * 70)
print("📄 文本摘要 Web 服务 - 摘要精灵")
print("=" * 70)

print("\n📚 正在加载摘要模型...")
# 使用支持中文的摘要模型
try:
    # 尝试使用中文摘要模型
    summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
    print("✅ 摘要精灵准备完毕！(多语言模型)")
except:
    print("⚠️  多语言模型加载失败,使用简化演示版本")
    summarizer = None
print("✅ 摘要精灵准备完毕！")

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
    <title>📄 文本摘要 - 摘要精灵</title>
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
            font-size: 26px;
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
                transform: translateY(100vh) rotate(360deg) scale(1.25);
                opacity: 0.25;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(0, 150, 136, 0.95) 0%, rgba(0, 121, 107, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0, 150, 136, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1000px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(0, 150, 136, 0.6);
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
            color: #b2dfdb;
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
            color: #00796b;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        textarea {{
            width: 100%;
            padding: 15px;
            border: 2px solid #009688;
            border-radius: 15px;
            font-size: 1em;
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            resize: vertical;
            min-height: 200px;
            transition: all 0.3s;
            line-height: 1.8;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: #00796b;
            box-shadow: 0 0 15px rgba(0, 150, 136, 0.3);
        }}
        
        .hint {{
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
        }}
        
        .length-control {{
            display: flex;
            gap: 15px;
            align-items: center;
            margin-top: 15px;
        }}
        
        .length-control label {{
            color: #00796b;
            font-weight: bold;
            margin: 0;
        }}
        
        .length-control input[type="range"] {{
            flex: 1;
            height: 8px;
            border-radius: 5px;
            background: #b2dfdb;
            outline: none;
        }}
        
        .length-control input[type="range"]::-webkit-slider-thumb {{
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #009688;
            cursor: pointer;
        }}
        
        .length-value {{
            color: #009688;
            font-weight: bold;
            font-size: 1.1em;
            min-width: 80px;
            text-align: right;
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
            box-shadow: 0 6px 20px rgba(0, 150, 136, 0.4);
            background: linear-gradient(135deg, #009688 0%, #00796b 100%);
            color: white;
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 150, 136, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(178, 223, 219, 0.95) 0%, rgba(128, 203, 196, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #009688;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 25px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .stat-item {{
            background: white;
            padding: 15px 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-label {{
            color: #00796b;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            color: #009688;
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .text-box {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #009688;
        }}
        
        .text-box h4 {{
            color: #00796b;
            margin-bottom: 12px;
            font-size: 1.2em;
        }}
        
        .text-content {{
            color: #555;
            line-height: 1.8;
            font-size: 1.05em;
        }}
        
        .summary-highlight {{
            background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #009688;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 文本摘要</h1>
        <p class="subtitle">摘要精灵帮你提炼文章精华！</p>
        
        <div class="input-area">
            <div class="input-group">
                <label>📝 输入长文本（中文/英文）：</label>
                <textarea id="inputText" placeholder="请输入需要摘要的文本...&#10;&#10;支持中文和英文的新闻文章、论文摘要、报告等各类长文本"></textarea>
                <div class="hint">
                    💡 提示：输入至少 50 字的文本，效果更佳（支持中文和英文）
                </div>
            </div>
            
            <div class="length-control">
                <label>📏 摘要长度：</label>
                <input type="range" id="lengthSlider" min="30" max="200" value="100" step="10">
                <span class="length-value" id="lengthValue">100 词</span>
            </div>
            
            <button id="summarizeBtn" onclick="summarize()" style="margin-top: 20px;">
                ✨ 生成摘要
            </button>
        </div>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        const fallingItems = ['📄', '📃', '📰', '📑', '📚', '📖', '📝', '✍️', '✨', '⭐', '🌟', '💫', '📊', '📈', '🔖', '📌'];
        
        function createFallingItem() {{
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 16 + 20) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => item.remove(), 7000);
        }}
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {
            setTimeout(createFallingItem, i * 150);
        }
        
        setInterval(createFallingItem, 150);
        
        // 更新长度显示
        document.getElementById('lengthSlider').addEventListener('input', function() {{
            document.getElementById('lengthValue').textContent = this.value + ' 词';
        }});
        
        async function summarize() {{
            const inputText = document.getElementById('inputText').value.trim();
            const maxLength = parseInt(document.getElementById('lengthSlider').value);
            
            if (!inputText) {{
                alert('请输入要摘要的文本！');
                return;
            }}
            
            const textLength = inputText.length;
            if (textLength < 50) {{
                alert('文本太短了！请输入至少 50 字的文本。');
                return;
            }}
            
            const resultDiv = document.getElementById('result');
            const summarizeBtn = document.getElementById('summarizeBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #00796b; font-size: 1.2em;">✨ 摘要精灵正在提炼精华...</p>';
            resultDiv.style.display = 'block';
            summarizeBtn.disabled = true;
            
            try {{
                const response = await fetch('/summarize', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ 
                        text: inputText,
                        max_length: maxLength
                    }})
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResult(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ 摘要失败: ${{error.message}}</p>`;
            }} finally {{
                summarizeBtn.disabled = false;
            }}
        }}
        
        function displayResult(data) {{
            const originalWords = data.original_length;
            const summaryWords = data.summary_length;
            const compression = ((1 - summaryWords / originalWords) * 100).toFixed(1);
            
            let html = '<h3 style="color: #00796b; margin-bottom: 20px; text-align: center;">✨ 摘要结果</h3>';
            
            html += '<div class="stats">';
            html += `
                <div class="stat-item">
                    <div class="stat-label">原文字数</div>
                    <div class="stat-value">${{originalWords}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">摘要字数</div>
                    <div class="stat-value">${{summaryWords}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">压缩率</div>
                    <div class="stat-value">${{compression}}%</div>
                </div>
            `;
            html += '</div>';
            
            html += '<div class="text-box summary-highlight">';
            html += '<h4>📌 摘要内容：</h4>';
            html += `<div class="text-content">${{data.summary}}</div>`;
            html += '</div>';
            
            html += '<div class="text-box">';
            html += '<h4>📄 原文内容：</h4>';
            html += `<div class="text-content">${{data.original.substring(0, 500)}}${{data.original.length > 500 ? '...' : ''}}</div>`;
            html += '</div>';
            
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

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data.get('text', '')
        max_length = data.get('max_length', 100)
        
        if not text:
            return jsonify({'error': '请输入文本'}), 400
        
        # 生成摘要
        if summarizer:
            min_length = max(20, max_length // 2)
            result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            summary = result[0]['summary_text']
        else:
            # 简化的中文摘要:取前几句话
            sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
            sentences = [s.strip() for s in sentences if s.strip()]
            # 取前3句或前100字
            summary_sentences = []
            total_len = 0
            for sent in sentences[:5]:
                if total_len + len(sent) <= max_length:
                    summary_sentences.append(sent)
                    total_len += len(sent)
                else:
                    break
            summary = ''.join(summary_sentences) if summary_sentences else sentences[0][:max_length]
            summary += " (简化摘要)"
        
        return jsonify({
            'original': text,
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary)
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
    print("📄 启动摘要精灵...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:7004")
    print("✨ 摘要精灵在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:7004')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=7004, debug=False)
