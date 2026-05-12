@echo off
setlocal
cd /d %~dp0

echo [LiteRaceSegNet V3 CURRENT ARCHIVE SUMMARY]
python seg\tools\current_archive_evidence_summary.py

if errorlevel 1 (
  echo [FAILED] archive summary failed.
  pause
  exit /b 1
)

echo [OK] final_evidence\current_run_archive\CURRENT_ARCHIVE_EVIDENCE_SUMMARY_V3.md
pause
