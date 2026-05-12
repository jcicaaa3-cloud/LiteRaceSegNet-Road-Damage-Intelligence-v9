@echo off
setlocal
cd /d %~dp0

set MASK_DIR=seg\runs\literace_upgrade_conservative\strict_val_pred\04_postprocessed_masks
set OUT_CSV=seg\runs\literace_upgrade_conservative\strict_val_pred\mask_quality_audit.csv

python seg\tools\audit_prediction_masks.py --mask_dir "%MASK_DIR%" --out_csv "%OUT_CSV%"

if errorlevel 1 (
  echo [FAILED] mask audit failed.
  pause
  exit /b 1
)

echo [OK] audit csv: %OUT_CSV%
pause
