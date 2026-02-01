@echo off
chcp 65001 >nul

REM 切换到脚本所在目录
cd /d "%~dp0"

echo ============================================================
echo 使用训练好的模型进行预测
echo ============================================================
echo.
echo 当前目录: %CD%
echo.

REM 检查模型文件夹是否存在
if not exist "my_sentiment_model" (
    echo ❌ 错误：找不到模型文件夹 my_sentiment_model
    echo.
    echo 请先运行 训练模型.bat 训练模型
    echo.
    pause
    exit /b 1
)

echo ✓ 模型文件夹存在
echo.

D:\aaaalokda\envs\myenv\python.exe 使用训练好的模型.py

pause
