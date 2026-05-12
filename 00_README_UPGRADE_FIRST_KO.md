# LiteRaceSegNet 업그레이드 ZIP 안내

이 ZIP의 목적은 결과 이미지를 고르는 것이 아니라, 다음 실행에서 과검출을 줄이기 위한 레포 업그레이드입니다.

## 핵심 변경

1. `seg/infer_literace_research_strict.py` 추가
   - mock/demo/fallback 마스크 없음
   - raw argmax, probability threshold, postprocess 결과를 따로 저장
   - 결과를 바꾸면 어느 단계에서 바뀌었는지 CSV로 남김

2. `seg/config/pothole_binary_literace_upgrade_conservative.yaml` 추가
   - 기존 damage class 과가중을 줄임
   - image size를 `[256, 384]`로 올림
   - brightness/contrast/noise augmentation 추가
   - validation/inference threshold `0.60` 기준 추가
   - 작은 잡음 component 제거 기준 `min_area_pixels: 120` 사용

3. `seg/train_literace.py` 업그레이드
   - 보수형 augmentation 설정을 config에서 읽음
   - validation에서 binary argmax 대신 softmax threshold를 사용할 수 있음

4. `seg/capstone_batch_service.py` 수정
   - 모델 모드에서 자동 demo fallback을 기본으로 꺼둠
   - fallback을 쓰려면 `--allow_demo_fallback`을 명시해야 함
   - 연구/포폴 증거용 결과와 UI 데모용 결과를 분리

5. 품질 점검 도구 추가
   - `seg/tools/audit_prediction_masks.py`
   - `seg/tools/evaluate_binary_segmentation.py`
   - `seg/tools/make_contact_sheet.py`

## 실행 순서

Windows:

```bat
03C_TRAIN_LITERACE_UPGRADE_CONSERVATIVE.bat
05B_INFER_LITERACE_STRICT_UPGRADED.bat
12_AUDIT_UPGRADED_MASKS.bat
```

Linux/AWS:

```bash
bash scripts/run_literace_upgrade_conservative.sh
bash scripts/run_literace_strict_infer_upgraded.sh
```

## 연구 정직성 기준

- 기존 결과 이미지는 조작하지 않습니다.
- 새 스크립트는 raw 예측, threshold 예측, postprocess 예측을 분리해서 저장합니다.
- postprocess는 성능 개선 후보이며, 논문/포폴 표에는 raw와 postprocess를 따로 적어야 합니다.
- demo fallback은 UI 흐름 확인용이며, 연구 성능 주장에 사용하지 않습니다.
