@echo off
setlocal
cd /d %~dp0

echo [LiteRaceSegNet UPGRADE TRAIN]
echo Conservative config: reduce over-detection, keep architecture claim unchanged.
python seg\train_literace.py --config seg\config\pothole_binary_literace_upgrade_conservative.yaml

if errorlevel 1 (
  echo [FAILED] upgrade training failed.
  pause
  exit /b 1
)

echo [OK] checkpoint: seg\runs\literace_upgrade_conservative\best.pth
pause
