@echo off
echo Setting up environment for discrete GPU training...

REM Force PyTorch to use discrete GPU
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF%

echo Starting training with GPU 0 (RTX 3050)...
D:/Baby-Cry-Model/.venv/Scripts/python.exe train_model.py

pause