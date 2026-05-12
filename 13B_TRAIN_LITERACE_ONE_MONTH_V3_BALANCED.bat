@echo off
setlocal
cd /d %~dp0

echo [LiteRaceSegNet ONE-MONTH V3 BALANCED FALLBACK]
echo Use this if the precision profile misses too many small potholes.
python seg\train_literace.py --config seg\config\pothole_binary_literace_one_month_v3_balanced.yaml

if errorlevel 1 (
  echo [FAILED] V3 balanced training failed.
  pause
  exit /b 1
)

echo [OK] checkpoint: seg\runs\literace_one_month_v3_balanced\best.pth
pause
