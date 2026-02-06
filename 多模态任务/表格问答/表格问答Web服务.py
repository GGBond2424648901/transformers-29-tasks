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
import numpy as np

# Monkey patch: 修复TAPAS tokenizer与pandas Arrow backend的兼容性问题
def patch_tapas_tokenizer():
    """修复TAPAS tokenizer处理DataFrame时的类型问题"""
    try:
        from transformers.models.tapas import tokenization_tapas
        
        # Patch 1: tokenize方法
        original_tokenize = tokenization_tapas.TapasTokenizer.tokenize
        
        def patched_tokenize(self, text, **kwargs):
            if not isinstance(text, str):
                text = str(text)
            return original_tokenize(self, text, **kwargs)
        
        tokenization_tapas.TapasTokenizer.tokenize = patched_tokenize
        
        # Patch 2: add_numeric_table_values函数 - 这是关键！
        original_add_numeric = tokenization_tapas.add_numeric_table_values
        
        def patched_add_numeric_table_values(table):
            # 先转换DataFrame为纯Python对象，避免Arrow backend
            import pandas as pd
            if isinstance(table, pd.DataFrame):
                # 重置索引，确保使用默认的RangeIndex
                table = table.reset_index(drop=True)
                # 创建一个新的DataFrame，使用Python原生类型
                new_data = {}
                for col in table.columns:
                    # 转换为列表，然后转为字符串
                    col_values = []
                    for val in table[col]:
                        col_values.append(str(val))
                    new_data[str(col)] = col_values
                
                # 用object dtype创建新DataFrame
                table = pd.DataFrame(new_data, dtype=object)
            
            return original_add_numeric(table)
        
        tokenization_tapas.add_numeric_table_values = patched_add_numeric_table_values
        
        print("✅ TAPAS tokenizer补丁已应用（包含DataFrame类型修复）")
    except Exception as e:
        print(f"⚠️  TAPAS tokenizer补丁应用失败: {e}")
        import traceback
        traceback.print_exc()

patch_tapas_tokenizer()

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
                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <input type="file" id="fileUpload" accept=".csv" style="display: none;">
                    <button class="quick-btn" onclick="document.getElementById('fileUpload').click()" style="padding: 10px 20px;">
                        📁 上传CSV文件
                    </button>
                    <span id="fileName" style="color: #00838f; line-height: 40px;"></span>
                </div>
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
                <input type="text" id="questionInput" placeholder="例如：What is the average salary? 或 How many employees are there?">
                <div class="quick-questions">
                    <button class="quick-btn" onclick="setQuestion('How many employees are there?')">有多少人</button>
                    <button class="quick-btn" onclick="setQuestion('What is the average salary?')">平均工资</button>
                    <button class="quick-btn" onclick="setQuestion('Who has the highest salary?')">最高工资</button>
                    <button class="quick-btn" onclick="setQuestion('What is the total salary?')">工资总和</button>
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
        
        // 文件上传处理
        document.getElementById('fileUpload').addEventListener('change', async function(e) {{
            const file = e.target.files[0];
            if (!file) return;
            
            document.getElementById('fileName').textContent = '正在读取: ' + file.name;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {{
                const response = await fetch('/upload', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    const textarea = document.getElementById('tableInput');
                    const originalHeader = textarea.value.split('\\n')[0] || '';
                    textarea.value = data.content;
                    const newHeader = data.content.split('\\n')[0] || '';
                    
                    // 检查是否进行了列名翻译
                    if (originalHeader && newHeader && originalHeader !== newHeader && 
                        originalHeader.match(/[\u4e00-\u9fff]/)) {{
                        document.getElementById('fileName').textContent = '✅ 已加载并自动翻译列名: ' + file.name;
                        document.getElementById('fileName').style.fontWeight = 'bold';
                    }} else {{
                        document.getElementById('fileName').textContent = '✅ 已加载: ' + file.name;
                    }}
                }} else {{
                    alert('文件读取失败: ' + data.error);
                    document.getElementById('fileName').textContent = '';
                }}
            }} catch (error) {{
                alert('上传失败: ' + error.message);
                document.getElementById('fileName').textContent = '';
            }}
        }});
        
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
                    let errorHtml = `<p style="text-align: center; color: #d32f2f; font-weight: bold;">❌ ${{data.error}}</p>`;
                    
                    // 如果有详细信息和解决方案，显示它们
                    if (data.details) {{
                        errorHtml += `<p style="text-align: center; color: #666; margin-top: 10px;">${{data.details}}</p>`;
                    }}
                    
                    if (data.solutions && data.solutions.length > 0) {{
                        errorHtml += '<div style="background: #fff3cd; padding: 15px; border-radius: 10px; margin-top: 15px; text-align: left;">';
                        errorHtml += '<h4 style="color: #856404; margin-bottom: 10px;">💡 解决方案：</h4>';
                        errorHtml += '<ul style="color: #856404; margin-left: 20px;">';
                        data.solutions.forEach(solution => {{
                            errorHtml += `<li style="margin: 5px 0;">${{solution}}</li>`;
                        }});
                        errorHtml += '</ul></div>';
                    }}
                    
                    resultDiv.innerHTML = errorHtml;
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

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 读取CSV文件
        content = file.read().decode('utf-8')
        
        # 尝试自动翻译中文列名
        lines = content.strip().split('\n')
        if lines:
            header = lines[0]
            # 常见中文列名映射
            column_mapping = {
                '姓名': 'Name',
                '名字': 'Name',
                '年龄': 'Age',
                '部门': 'Department',
                '工资': 'Salary',
                '薪资': 'Salary',
                '薪水': 'Salary',
                '职位': 'Position',
                '入职日期': 'Join_Date',
                '入职时间': 'Join_Date',
                '性别': 'Gender',
                '电话': 'Phone',
                '邮箱': 'Email',
                '地址': 'Address',
                '城市': 'City',
                '省份': 'Province',
                '国家': 'Country',
                '产品': 'Product',
                '价格': 'Price',
                '数量': 'Quantity',
                '总额': 'Total',
                '收入': 'Revenue',
                '月份': 'Month',
                '日期': 'Date',
                '时间': 'Time',
                '类别': 'Category',
                '状态': 'Status',
                '备注': 'Note',
            }
            
            # 检查是否有中文列名
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in header)
            
            if has_chinese:
                # 翻译列名
                columns = header.split(',')
                translated_columns = []
                for col in columns:
                    col = col.strip()
                    if col in column_mapping:
                        translated_columns.append(column_mapping[col])
                    else:
                        # 如果没有映射，保持原样
                        translated_columns.append(col)
                
                # 重建CSV内容
                new_header = ','.join(translated_columns)
                lines[0] = new_header
                content = '\n'.join(lines)
                
                print(f"✅ 自动翻译列名: {header} -> {new_header}")
        
        return jsonify({'success': True, 'content': content})
        
    except Exception as e:
        return jsonify({'error': f'文件读取失败: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        table_text = data.get('table', '')
        question = data.get('question', '')
        
        if not table_text or not question:
            return jsonify({'error': '请提供表格和问题'}), 400
        
        # 解析CSV - 使用最简单的方法
        from io import StringIO
        
        # 直接读取为object类型，然后立即转换所有值为字符串
        df = pd.read_csv(StringIO(table_text), dtype=str, keep_default_na=False)
        
        if df.empty:
            return jsonify({'error': '表格数据为空，请检查输入格式'}), 400
        
        # 确保索引是RangeIndex
        df = df.reset_index(drop=True)
        
        print(f"解析的表格:\n{df}")
        print(f"表格shape: {df.shape}")
        print(f"表格columns: {list(df.columns)}")
        print(f"表格index: {df.index}")
        print(f"表格数据类型:\n{df.dtypes}")
        print(f"问题: {question}")
        
        # 查询
        result = table_qa(table=df, query=question)
        print(f"查询结果: {result}")
        
        # 处理答案，让它更友好
        answer = result['answer']
        aggregator = result.get('aggregator', 'NONE')
        cells = result.get('cells', [])
        coordinates = result.get('coordinates', [])
        
        # 根据聚合类型优化答案显示
        if aggregator == 'COUNT':
            # 统计数量：显示数字和名单
            if len(cells) <= 10:
                answer = f"{len(cells)} 人：{', '.join(cells)}"
            else:
                answer = f"{len(cells)} 人：{', '.join(cells[:10])} ..."
        elif aggregator == 'SUM':
            # 求和：尝试计算数值总和
            try:
                # 检查是否是数字列
                numeric_cells = []
                for c in cells:
                    c_clean = str(c).replace(',', '').replace(' ', '')
                    if c_clean.replace('.', '').replace('-', '').isdigit():
                        numeric_cells.append(float(c_clean))
                
                if numeric_cells:
                    # 是数字，计算总和
                    total = sum(numeric_cells)
                    answer = f"{total:,.2f}"
                else:
                    # 不是数字（比如姓名），显示数量和名单
                    if len(cells) <= 10:
                        answer = f"{len(cells)} 人：{', '.join(cells)}"
                    else:
                        answer = f"{len(cells)} 人：{', '.join(cells[:10])} ..."
            except:
                # 出错时显示数量
                if len(cells) <= 10:
                    answer = f"{len(cells)} 项：{', '.join(cells)}"
                else:
                    answer = f"{len(cells)} 项"
        elif aggregator == 'AVERAGE':
            # 平均值：计算数值平均
            try:
                numeric_cells = []
                for c in cells:
                    c_clean = str(c).replace(',', '').replace(' ', '')
                    if c_clean.replace('.', '').replace('-', '').isdigit():
                        numeric_cells.append(float(c_clean))
                
                if numeric_cells:
                    avg = sum(numeric_cells) / len(numeric_cells)
                    answer = f"{avg:,.2f}"
                else:
                    answer = str(cells[0]) if cells else answer
            except:
                answer = str(cells[0]) if cells else answer
        elif aggregator == 'NONE':
            # 无聚合：直接显示答案（通常是单个值）
            if isinstance(answer, str) and answer.startswith('NONE > '):
                answer = answer.replace('NONE > ', '')
            # 如果答案太长，只显示前几个
            if len(cells) > 5:
                answer = ', '.join(cells[:5]) + f' ... (共{len(cells)}项)'
            elif cells:
                answer = ', '.join(cells)
        
        # 生成表格HTML
        table_html = df.to_html(index=False, classes='data-table')
        
        return jsonify({
            'question': question,
            'answer': answer,
            'table_html': table_html
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"错误详情: {error_details}")
        
        # 检查是否是已知的pandas Arrow backend问题
        error_msg = str(e)
        if "iteration over a 0-d array" in error_msg or "KeyError: 0" in error_details:
            return jsonify({
                'error': '表格解析失败：pandas Arrow backend兼容性问题',
                'details': '这是pandas 2.x与TAPAS模型的已知兼容性问题。',
                'solutions': [
                    '1. 使用命令行脚本：python 表格问答示例.py',
                    '2. 降级pandas版本：pip install "pandas<2.0.0"',
                    '3. 使用英文表格数据可能效果更好',
                    '4. 查看"已知问题说明.md"了解详情'
                ]
            }), 500
        
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
