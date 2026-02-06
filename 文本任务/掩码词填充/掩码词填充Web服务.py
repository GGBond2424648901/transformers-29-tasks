#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码词填充 Web 服务 - 填词魔法师 🎭
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify
from transformers import pipeline
import base64

BACKGROUND_PATH = r'背景.png'

print("=" * 70)
print("🎭 掩码词填充 Web 服务 - 填词魔法师")
print("=" * 70)

print("\n🔮 正在加载填词模型...")
fill_mask = pipeline("fill-mask", model="bert-base-chinese")
print("✅ 填词魔法师准备完毕！")

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
    <title>🎭 掩码词填充 - 填词魔法师</title>
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
            font-size: 28px;
            animation: fall linear infinite;
            z-index: 1;
            pointer-events: none;
            opacity: 0.75;
        }}
        
        @keyframes fall {{
            0% {{
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.75;
            }}
            100% {{
                transform: translateY(100vh) rotate(360deg) scale(1.3);
                opacity: 0.2;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(255, 87, 34, 0.95) 0%, rgba(244, 67, 54, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(255, 87, 34, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 900px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(255, 87, 34, 0.6);
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
            color: #ffe0b2;
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
            color: #e64a19;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        textarea {{
            width: 100%;
            padding: 15px;
            border: 2px solid #ff5722;
            border-radius: 15px;
            font-size: 1.1em;
            font-family: 'Microsoft YaHei', 'Arial', sans-serif;
            resize: vertical;
            min-height: 120px;
            transition: all 0.3s;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: #e64a19;
            box-shadow: 0 0 15px rgba(255, 87, 34, 0.3);
        }}
        
        .hint {{
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
            line-height: 1.6;
        }}
        
        .example-box {{
            background: #fff3e0;
            padding: 12px;
            border-radius: 10px;
            margin-top: 10px;
            border-left: 4px solid #ff5722;
        }}
        
        .example-box code {{
            color: #e64a19;
            font-weight: bold;
            font-size: 1.05em;
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
            box-shadow: 0 6px 20px rgba(255, 87, 34, 0.4);
            background: linear-gradient(135deg, #ff5722 0%, #e64a19 100%);
            color: white;
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 87, 34, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(255, 224, 178, 0.95) 0%, rgba(255, 204, 188, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #ff5722;
        }}
        
        .result-item {{
            background: white;
            padding: 18px;
            border-radius: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #ff5722;
            transition: all 0.3s;
        }}
        
        .result-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(255, 87, 34, 0.2);
        }}
        
        .result-rank {{
            display: inline-block;
            background: linear-gradient(135deg, #ff5722 0%, #e64a19 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            margin-right: 10px;
            font-size: 0.9em;
        }}
        
        .result-word {{
            color: #e64a19;
            font-weight: bold;
            font-size: 1.3em;
            margin: 10px 0;
        }}
        
        .result-sentence {{
            color: #555;
            font-size: 1.05em;
            line-height: 1.8;
            margin: 10px 0;
        }}
        
        .result-score {{
            color: #ff5722;
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #ffccbc;
            border-radius: 4px;
            margin-top: 8px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff5722 0%, #e64a19 100%);
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 掩码词填充</h1>
        <p class="subtitle">填词魔法师帮你预测缺失的词语！</p>
        
        <div class="input-area">
            <div class="input-group">
                <label>📝 输入带掩码的文本：</label>
                <textarea id="inputText" placeholder="请输入包含 [MASK] 的中文句子..."></textarea>
                <div class="hint">
                    💡 提示：使用 <code>[MASK]</code> 标记需要填充的位置
                </div>
                <div class="example-box">
                    <strong>示例：</strong><br>
                    <code>今天天气真[MASK]，适合出去玩。</code><br>
                    <code>我喜欢在[MASK]里看书。</code><br>
                    <code>这部电影[MASK]好看，我推荐大家去看。</code>
                </div>
            </div>
            
            <button id="fillBtn" onclick="fillMask()">
                🔮 开始填词
            </button>
        </div>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        const fallingItems = ['🎭', '🎪', '🎨', '✨', '⭐', '🌟', '💫', '🔮', '📝', '✏️', '📖', '📚', '🎯', '🎲', '🎰', '🎴'];
        
        function createFallingItem() {{
            const item = document.createElement('div');
            item.className = 'falling-item';
            item.textContent = fallingItems[Math.floor(Math.random() * fallingItems.length)];
            item.style.left = Math.random() * 100 + '%';
            item.style.animationDuration = (Math.random() * 3 + 4) + 's';
            item.style.fontSize = (Math.random() * 18 + 22) + 'px';
            document.body.appendChild(item);
            
            setTimeout(() => item.remove(), 7000);
        }}
        
        // 初始创建10个飘落元素
        for(let i = 0; i < 10; i++) {{
            setTimeout(createFallingItem, i * 150);
        }}
        
        setInterval(createFallingItem, 150);
        
        async function fillMask() {{
            const inputText = document.getElementById('inputText').value.trim();
            
            if (!inputText) {{
                alert('请输入带掩码的文本！');
                return;
            }}
            
            if (!inputText.includes('[MASK]')) {{
                alert('请在文本中使用 [MASK] 标记需要填充的位置！');
                return;
            }}
            
            const resultDiv = document.getElementById('result');
            const fillBtn = document.getElementById('fillBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #e64a19; font-size: 1.2em;">🔮 填词魔法师正在思考...</p>';
            resultDiv.style.display = 'block';
            fillBtn.disabled = true;
            
            try {{
                const response = await fetch('/fill', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ text: inputText }})
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResult(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ 填词失败: ${{error.message}}</p>`;
            }} finally {{
                fillBtn.disabled = false;
            }}
        }}
        
        function displayResult(data) {{
            let html = '<h3 style="color: #e64a19; margin-bottom: 20px; text-align: center;">✨ 填词结果（按可能性排序）</h3>';
            
            data.results.forEach((item, index) => {{
                const percentage = (item.score * 100).toFixed(1);
                html += `
                    <div class="result-item">
                        <div>
                            <span class="result-rank">Top ${{index + 1}}</span>
                            <span class="result-score">置信度: ${{percentage}}%</span>
                        </div>
                        <div class="result-word">填入词语: ${{item.token_str}}</div>
                        <div class="result-sentence">"${{item.sequence}}"</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${{percentage}}%"></div>
                        </div>
                    </div>
                `;
            }});
            
            document.getElementById('result').innerHTML = html;
        }}
        
        // 回车键填词
        document.getElementById('inputText').addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'Enter') {{
                fillMask();
            }}
        }});
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

@app.route('/fill', methods=['POST'])
def fill():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '请输入文本'}), 400
        
        if '[MASK]' not in text:
            return jsonify({'error': '文本中必须包含 [MASK] 标记'}), 400
        
        # 填充掩码
        results = fill_mask(text)
        
        # 确保返回列表格式
        if not isinstance(results, list):
            results = [results]
        
        # 转换numpy类型为Python原生类型
        for result in results:
            if isinstance(result, dict) and 'score' in result:
                result['score'] = float(result['score'])
        
        return jsonify({
            'original': text,
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
    print("🎭 启动填词魔法师...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:7003")
    print("🔮 填词魔法师在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:7003')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=7003, debug=False)
