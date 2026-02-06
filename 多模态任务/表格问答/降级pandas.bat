@echo off
chcp 65001 >nul
echo ========================================
echo 降级 pandas 到 1.5.3
echo ========================================
echo.
echo 当前正在降级 pandas 版本以解决 TAPAS 兼容性问题...
echo.

D:\aaaalokda\envs\myenv\python.exe -m pip install "pandas==1.5.3" --force-reinstall

echo.
echo ========================================
echo 降级完成！
echo ========================================
echo.
echo 验证 pandas 版本：
D:\aaaalokda\envs\myenv\python.exe -c "import pandas; print(f'pandas 版本: {pandas.__version__}')"
echo.
pause
