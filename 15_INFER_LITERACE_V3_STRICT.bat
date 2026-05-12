@echo off
setlocal
cd /d %~dp0

echo [LiteRaceSegNet V3 STRICT INFERENCE]
echo Put test images in assets\service_demo\input_batch first.
python seg\infer_literace_research_strict.py ^
  --input_dir assets\service_demo\input_batch ^
  --config seg\config\pothole_binary_literace_one_month_v3.yaml ^
  --ckpt seg\runs\literace_one_month_v3\best.pth ^
  --outdir seg\runs\literace_one_month_v3_strict_infer ^
  --threshold 0.65 ^
  --min_area_pixels 180 ^
  --save_prob

if errorlevel 1 (
  echo [FAILED] V3 strict inference failed.
  pause
  exit /b 1
)

echo [OK] seg\runs\literace_one_month_v3_strict_infer
pause
