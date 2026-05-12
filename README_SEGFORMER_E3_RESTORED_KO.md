# V6 SegFormer-E3 복구 패치

V6에서 빠졌던 SegFormer-E3 baseline 경로를 다시 넣는 패치입니다. 기존 LiteRaceSegNet V6 파일은 지우지 않고, SegFormer-E3 학습/평가/시각화/evidence 회수 스크립트만 추가합니다.

## 적용

```bash
cd /home/ubuntu/road-damage-segmentation-portfolio
unzip -o ~/LRS_V6_SEGFORMER_E3_RESTORE_PATCH.zip -d .
```

## SegFormer-E3 실행

```bash
bash scripts/run_v6_segformer_e3_baseline.sh
```

결과:

```text
LRS_SEGFORMER_E3_V6_EVIDENCE.zip
```

## OOM 뜨면

```bash
bash scripts/run_v6_segformer_e3_safe.sh
```

## 새 val 준비부터 LiteRace V6 + SegFormer-E3까지 한 번에

```bash
bash scripts/run_v6_literace_plus_segformer_e3.sh ~/LRS_NEW_VAL_10_PAIRS.zip
```

## 보고서 비교표

```text
V3 / V4-A / V5-A / V6-A / SegFormer-E3
mIoU / damage IoU / precision / recall / dice / model size / latency
```
