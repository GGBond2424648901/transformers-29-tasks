@echo off
chcp 65001 >nul
echo ======================================================================
echo 🚀 ChatGLM-6B LoRA 微调 - 智能客服
echo ======================================================================
echo.

cd /d "%~dp0"

echo 📍 当前目录: %CD%
echo.

echo 🐍 使用 Python 环境: D:\aaaalokda\envs\myenv\python.exe
echo.

echo ⚙️  开始训练...
echo.

D:\aaaalokda\envs\myenv\python.exe chatglm_lora_finetune.py

echo.
echo ======================================================================
echo ✨ 训练完成！
echo ======================================================================
echo.
echo 下一步：
echo 1. 运行 test_model.py 测试模型
echo 2. 双击 启动客服系统.bat 启动 Web 服务
echo.

pause
