#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格问答 Web 服务 - 表格智者 📊
"""

import os
os.environ['HF_HOME'] = r'D:\transformers训练\transformers-main\预训练模型下载处'
os.environ['TRANSFORMERS_CACHE'] = r'D:\transformers训练\transformers-main\预训练模型下载处'

from flask import Flask, request, jsonify, send_file
from transformers import pipeline
import pandas as pd
import base64

BACKGROUND_PATH = r'背景.png'

print("=" * 70)
print("📊 表格问答 Web 服务 - 表格智者")
print("=" * 70)

print("\n📚 正在加载表格问答模型...")
table_qa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
print("✅ 表格智者准备完毕！")

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
    <title>📊 表格问答 - 表格智者</title>
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
            opacity: 0.73;
        }}
        
        @keyframes fall {{
            0% {{
                transform: translateY(-10px) rotate(0deg) scale(1);
                opacity: 0.73;
            }}
            100% {{
                transform: translateY(100vh) rotate(360deg) scale(1.3);
                opacity: 0.23;
            }}
        }}
        
        .container {{
            background: linear-gradient(135deg, rgba(0, 188, 212, 0.95) 0%, rgba(0, 151, 167, 0.95) 100%);
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0, 188, 212, 0.5);
            padding: 40px;
            max-width: 1200px;
            margin: 20px auto;
            max-width: 1200px;
            width: 100%;
            backdrop-filter: blur(10px);
            border: 3px solid rgba(0, 188, 212, 0.6);
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
            color: #b2ebf2;
            margin-bottom: 30px;
            font-size: 1.2em;
        }}
        
        .input-area {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
        }}
        
        .table-input {{
            margin-bottom: 25px;
        }}
        
        .table-input label {{
            display: block;
            color: #00838f;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        textarea {{
            width: 100%;
            padding: 15px;
            border: 2px solid #00bcd4;
            border-radius: 15px;
            font-size: 1em;
            font-family: 'Courier New', monospace;
            resize: vertical;
            min-height: 150px;
            transition: all 0.3s;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: #00838f;
            box-shadow: 0 0 15px rgba(0, 188, 212, 0.3);
        }}
        
        .hint {{
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
        }}
        
        .example-box {{
            background: #e0f7fa;
            padding: 12px;
            border-radius: 10px;
            margin-top: 10px;
            border-left: 4px solid #00bcd4;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        .question-input {{
            margin-bottom: 20px;
        }}
        
        .question-input input {{
            width: 100%;
            padding: 15px;
            border: 2px solid #00bcd4;
            border-radius: 15px;
            font-size: 1.05em;
            transition: all 0.3s;
        }}
        
        .question-input input:focus {{
            outline: none;
            border-color: #00838f;
            box-shadow: 0 0 15px rgba(0, 188, 212, 0.3);
        }}
        
        .quick-questions {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        
        .quick-btn {{
            padding: 8px 15px;
            background: linear-gradient(135deg, #26c6da 0%, #00bcd4 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }}
        
        .quick-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 188, 212, 0.4);
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
            box-shadow: 0 6px 20px rgba(0, 188, 212, 0.4);
            background: linear-gradient(135deg, #00bcd4 0%, #00838f 100%);
            color: white;
        }}
        
        button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 188, 212, 0.5);
        }}
        
        button:disabled {{
            background: #ddd;
            cursor: not-allowed;
            transform: none;
        }}
        
        .result-container {{
            background: linear-gradient(135deg, rgba(178, 235, 242, 0.95) 0%, rgba(128, 222, 234, 0.95) 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            display: none;
            border: 3px solid #00bcd4;
        }}
        
        .table-display {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: linear-gradient(135deg, #00bcd4 0%, #00838f 100%);
            color: white;
            font-weight: bold;
        }}
        
        tr:hover {{
            background: #e0f7fa;
        }}
        
        .answer-box {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            border-left: 4px solid #00bcd4;
        }}
        
        .question-text {{
            color: #00838f;
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        
        .answer-text {{
            color: #006064;
            font-size: 1.4em;
            font-weight: bold;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 表格问答</h1>
        <p class="subtitle">表格智者帮你查询表格数据！</p>
        
        <div class="input-area">
            <div class="table-input">
                <label>📋 输入表格数据（CSV格式）：</label>
                <textarea id="tableInput" placeholder="姓名,年龄,部门,工资&#10;张三,25,技术部,8000&#10;李四,30,销售部,9000&#10;王五,28,技术部,8500"></textarea>
                <div class="hint">
                    💡 提示：第一行为表头，使用逗号分隔列
                </div>
                <div class="example-box">
                    <strong>示例格式：</strong><br>
                    姓名,年龄,部门,工资<br>
                    张三,25,技术部,8000<br>
                    李四,30,销售部,9000
                </div>
            </div>
            
            <div class="question-input">
                <label>❓ 提出问题：</label>
                <input type="text" id="questionInput" placeholder="例如：技术部有多少人？">
                <div class="quick-questions">
                    <button class="quick-btn" onclick="setQuestion('How many people?')">人数</button>
                    <button class="quick-btn" onclick="setQuestion('What is the average?')">平均值</button>
                    <button class="quick-btn" onclick="setQuestion('Who has the highest?')">最高</button>
                    <button class="quick-btn" onclick="setQuestion('What is the total?')">总和</button>
                </div>
            </div>
            
            <button id="askBtn" onclick="askQuestion()">
                🔍 查询答案
            </button>
        </div>
        
        <div id="result" class="result-container"></div>
    </div>
    
    <script>
        const fallingItems = ['📊', '📈', '📉', '📋', '📑', '🔢', '💹', '📊', '🗂️', '📁', '✨', '⭐', '🌟', '💫', '🔍', '🔎'];
        
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
        
        function setQuestion(question) {{
            document.getElementById('questionInput').value = question;
        }}
        
        async function askQuestion() {{
            const tableText = document.getElementById('tableInput').value.trim();
            const question = document.getElementById('questionInput').value.trim();
            
            if (!tableText) {{
                alert('请输入表格数据！');
                return;
            }}
            
            if (!question) {{
                alert('请输入问题！');
                return;
            }}
            
            const resultDiv = document.getElementById('result');
            const askBtn = document.getElementById('askBtn');
            
            resultDiv.innerHTML = '<p style="text-align: center; color: #00838f; font-size: 1.2em;">🔍 表格智者正在查询...</p>';
            resultDiv.style.display = 'block';
            askBtn.disabled = true;
            
            try {{
                const response = await fetch('/ask', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ 
                        table: tableText,
                        question: question
                    }})
                }});
                
                const data = await response.json();
                
                if (data.error) {{
                    resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ ${{data.error}}</p>`;
                }} else {{
                    displayResult(data);
                }}
            }} catch (error) {{
                resultDiv.innerHTML = `<p style="text-align: center; color: #d32f2f;">❌ 查询失败: ${{error.message}}</p>`;
            }} finally {{
                askBtn.disabled = false;
            }}
        }}
        
        function displayResult(data) {{
            let html = '<h3 style="color: #00838f; margin-bottom: 20px; text-align: center;">✨ 查询结果</h3>';
            
            // 显示表格
            html += '<div class="table-display">';
            html += '<h4 style="color: #00838f; margin-bottom: 15px;">📋 数据表格：</h4>';
            html += data.table_html;
            html += '</div>';
            
            // 显示答案
            html += '<div class="answer-box">';
            html += '<div class="question-text">❓ ' + data.question + '</div>';
            html += '<div class="answer-text">💡 ' + data.answer + '</div>';
            html += '</div>';
            
            document.getElementById('result').innerHTML = html;
        }}
        
        document.getElementById('questionInput').addEventListener('keydown', function(e) {{
            if (e.key === 'Enter') {{
                askQuestion();
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

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        table_text = data.get('table', '')
        question = data.get('question', '')
        
        if not table_text or not question:
            return jsonify({'error': '请提供表格和问题'}), 400
        
        # 解析CSV
        from io import StringIO
        df = pd.read_csv(StringIO(table_text))
        
        # 查询
        result = table_qa(table=df, query=question)
        
        # 生成表格HTML
        table_html = df.to_html(index=False, classes='data-table')
        
        return jsonify({
            'question': question,
            'answer': result['answer'],
            'table_html': table_html
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
    print("📊 启动表格智者...")
    print("=" * 70)
    print("\n📍 访问地址: http://localhost:8002")
    print("🔍 表格智者在这里等你~\n")
    
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open('http://localhost:8002')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=8002, debug=False)
