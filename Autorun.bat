@echo off
cd /d "%~dp0"

echo ==================================================
echo Launching Dual TabbyAPI Stack
echo ==================================================
echo GPU 0: 12B Model (Port 5000 - RTX 4080 Super)
echo GPU 1: 26B Model (Port 5001 - RTX 4000 Ada)
echo ==================================================

:: Launch 12B Model on GPU 0 (Port 5000)
start "TabbyAPI - 12B (GPU 0 / Port 5000)" cmd /k "set CUDA_VISIBLE_DEVICES=0 && venv\Scripts\python.exe main.py --config config_4080S.yml"

:: Launch 26B Model on GPU 1 (Port 5001)
start "TabbyAPI - 26B (GPU 1 / Port 5001)" cmd /k "set CUDA_VISIBLE_DEVICES=1 && venv\Scripts\python.exe main.py --config config_ada4000.yml"

echo Startup commands issued for both TabbyAPI instances.