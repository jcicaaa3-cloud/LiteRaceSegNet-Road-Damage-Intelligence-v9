@echo off
setlocal

set CONFIG=seg\config\pothole_binary_literace_train.yaml
set CKPT=seg\runs\literace_boundary_degradation\best.pth
set INPUT_DIR=datasets\pothole_binary\processed\val\images
set OUTDIR=seg\runs\literace_boundary_degradation\real_pred_val

python seg\infer_literace_to_service.py --config "%CONFIG%" --ckpt "%CKPT%" --input_dir "%INPUT_DIR%" --outdir "%OUTDIR%"

endlocal
