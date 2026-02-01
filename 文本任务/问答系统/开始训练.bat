@echo off
chcp 65001 >nul

echo ============================================================
echo 问答系统训练
echo ============================================================
echo.

cd /d "%~dp0"

echo 当前目录: %CD%
echo.

echo 检查 Python 环境...
D:\aaaalokda\envs\myenv\python.exe --version
if errorlevel 1 (
    echo ❌ Python 环境未找到
    pause
    exit /b 1
)

echo.
echo 开始训练...
echo.

D:\aaaalokda\envs\myenv\python.exe 简单训练示例.py

echo.
echo ============================================================
echo 训练完成！
echo ============================================================
echo.

pause
