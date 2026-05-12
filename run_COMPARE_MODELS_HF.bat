@echo off
setlocal
cd /d %~dp0

echo ============================================================
echo [Compare] LiteRaceSegNet/CNN candidate vs HuggingFace SegFormer-03
echo CPU latency is measured for lightweight deployment evidence.
echo ============================================================
echo.

if not exist "seg\transformer_03\hf_pretrained\segformer_b3_ade\config.json" (
  echo [WARN] Local HuggingFace SegFormer folder was not found.
  echo Run 02_SETUP_SEGFORMER_03_HF.bat first.
  pause
  exit /b 1
)

python seg\compare\compare_models.py ^
 --configs seg\config\pothole_binary_literace_train.yaml seg\config\pothole_binary_segformer_03_hf.yaml ^
 --names LiteRaceSegNet_CNN SegFormer_03_HF_Transformer ^
 --device cpu ^
 --batch_size 1 ^
 --latency_warmup 10 ^
 --latency_repeats 50 ^
 --outdir seg\runs\model_compare_hf

pause
