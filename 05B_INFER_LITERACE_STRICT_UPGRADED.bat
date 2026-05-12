@echo off
setlocal
cd /d %~dp0

set CONFIG=seg\config\pothole_binary_literace_upgrade_conservative.yaml
set CKPT=seg\runs\literace_upgrade_conservative\best.pth
set INPUT_DIR=datasets\pothole_binary\processed\val\images
set OUTDIR=seg\runs\literace_upgrade_conservative\strict_val_pred

python seg\infer_literace_research_strict.py ^
 --config "%CONFIG%" ^
 --ckpt "%CKPT%" ^
 --input_dir "%INPUT_DIR%" ^
 --outdir "%OUTDIR%" ^
 --threshold 0.60 ^
 --min_area_pixels 120 ^
 --save_prob

if errorlevel 1 (
  echo [FAILED] strict inference failed.
  pause
  exit /b 1
)

echo [OK] strict inference result: %OUTDIR%
pause
