# LiteRaceSegNet One-Month V3 추가팩 안내

이 추가팩은 기존 레포, 기존 결과 archive, 기존 논문/보고서 틀을 덮어쓰지 않는다.  
현재 상태에서 한 달 더 진행할 때 가장 방어 가능한 개선 방향만 추가했다.

## 왜 V3인가

현재 archive 기준으로 LiteRaceSegNet은 손상 위치를 일부 포착하지만, 여러 validation/service 이미지에서 도로 질감과 그림자까지 손상으로 넓게 잡는 과검출이 남아 있다. 그래서 구조를 크게 바꾸는 대신 다음 네 가지 증거를 보강한다.

1. validation threshold sweep
2. min-area connected component filtering sweep
3. boundary/component quality metric
4. 한 달 연장 실험 계획 및 발표 방어 문장

## 기존 파일 보존 원칙

- 기존 `README.md`, 기존 `docs/reports/*.docx`, 기존 `final_evidence/current_run_archive/*`는 덮어쓰지 않는다.
- 새 파일은 `V3`, `one_month`, `threshold_sweep`, `boundary_component` 이름으로 분리했다.
- raw prediction, threshold prediction, postprocess prediction은 따로 저장해야 한다.
- postprocess 결과를 raw 성능처럼 말하면 안 된다.

## Windows 실행 순서

```bat
13_TRAIN_LITERACE_ONE_MONTH_V3.bat
14_SWEEP_THRESHOLD_MINAREA_V3.bat
15_INFER_LITERACE_V3_STRICT.bat
16_BOUNDARY_COMPONENT_METRICS_V3.bat
17_SUMMARIZE_CURRENT_ARCHIVE_V3.bat
```

precision profile이 너무 보수적이라 작은 포트홀을 놓치면 아래 fallback을 쓴다.

```bat
13B_TRAIN_LITERACE_ONE_MONTH_V3_BALANCED.bat
```

## Linux / AWS 실행 순서

```bash
bash scripts/run_literace_one_month_v3.sh
bash scripts/run_threshold_sweep_v3.sh
bash scripts/run_literace_v3_strict_infer.sh
bash scripts/run_boundary_component_metrics_v3.sh
python seg/tools/current_archive_evidence_summary.py
```

## 발표에서 잡을 포인트

- “성능이 낮다”가 아니라 “작은 데이터셋에서 과검출이 확인되어, 정량 지표와 마스크 품질 지표로 개선 방향을 분리했다.”라고 말한다.
- mIoU만 말하지 말고 Damage IoU, precision/recall, boundary F1, component count를 같이 본다.
- LiteRaceSegNet은 경량성/배포 가능성의 trade-off 모델이고, SegFormer-03은 전역 문맥 baseline으로 남긴다.
- 한 달 연장 목표는 논문 틀 변경이 아니라 evidence quality 향상이다.
