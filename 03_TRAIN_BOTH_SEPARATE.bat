@echo off
setlocal
cd /d %~dp0
set NO_PAUSE=1

echo ============================================================
echo [Train Both - Strictly Separated]
echo 1) LiteRaceSegNet ONLY
echo 2) SegFormer-03 ONLY
echo ============================================================
echo No mixed checkpoint. No shared output folder.
echo.

echo [STEP 1/2] LiteRaceSegNet training starts.
call 03A_TRAIN_LITERACESEGNET_ONLY.bat
if errorlevel 1 (
  echo [STOP] LiteRaceSegNet training failed. SegFormer will NOT start.
  pause
  exit /b 1
)

echo.
echo [STEP 2/2] SegFormer-03 training starts.
call 03B_TRAIN_SEGFORMER_03_ONLY.bat
if errorlevel 1 (
  echo [STOP] SegFormer-03 training failed.
  pause
  exit /b 1
)

echo.
echo [OK] Both models trained separately.
echo LiteRaceSegNet: seg\runs\literace_boundary_degradation\best.pth
echo SegFormer-03: seg\transformer_03\checkpoints\segformer_03_best.pth
echo.
echo Next: 04_COMPARE_AFTER_SEGFORMER_03_TRAIN.bat
pause
