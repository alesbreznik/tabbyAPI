@echo off
cd /d "%~dp0\tabbyAPI-gradio-loader"
call venv\Scripts\activate
python webui.py
pause
