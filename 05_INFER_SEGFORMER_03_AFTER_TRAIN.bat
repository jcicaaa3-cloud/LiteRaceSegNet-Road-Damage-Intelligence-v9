@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [Infer] Run fine-tuned SegFormer-03 on service demo images
echo ============================================================
echo.

if not exist "seg\transformer_03\checkpoints\segformer_03_best.pth" (
  echo [ERROR] Fine-tuned SegFormer checkpoint not found.
  echo Run 03_TRAIN_SEGFORMER_03_POTHOLE.bat first.
  pause
  exit /b 1
)

python seg\infer_seg.py ^
 --config seg\config\pothole_binary_segformer_03_train.yaml ^
 --ckpt seg\transformer_03\checkpoints\segformer_03_best.pth ^
 --input_dir assets\service_demo\input_batch ^
 --output_dir seg\runs\segformer_03_infer_after_train

pause
