@echo off
setlocal
cd /d %~dp0

echo [LiteRaceSegNet ONE-MONTH V3 TRAIN]
echo Existing paper/report structure is not modified. This only runs the added V3 config.
python seg\train_literace.py --config seg\config\pothole_binary_literace_one_month_v3.yaml

if errorlevel 1 (
  echo [FAILED] V3 training failed. Try the balanced fallback script.
  pause
  exit /b 1
)

echo [OK] checkpoint: seg\runs\literace_one_month_v3\best.pth
pause
