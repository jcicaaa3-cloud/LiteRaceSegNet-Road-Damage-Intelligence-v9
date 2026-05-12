# LiteRaceSegNet V5 Update Patch

이 패치는 새 데이터셋을 다시 보내는 게 아니라, 기존 서버에 이미 풀린 데이터셋/증강 데이터셋을 그대로 사용해서 config와 실행 스크립트만 업데이트합니다.

## 왜 업데이트하나

이번 V4 결과 로그를 보면 best는 좋아졌지만 이상한 점이 있습니다.

```text
best epoch = 2
best mIoU = 0.643803
best damage IoU = 0.445210
last mIoU = 0.608268
last damage IoU = 0.379589
```

그리고 config에는 `optimizer.lr: 0.00040`이 있었지만 실제 로그의 첫 LR은 약 `0.001`에서 시작했습니다.
기존 학습 config 형식은 `train.base_lr`, `train.class_weights`, `train.loss`를 읽는 구조라서, V4 config의 일부 키가 의도대로 반영되지 않았을 가능성이 큽니다.

그래서 V5는 다음을 고칩니다.

- `optimizer.lr` → `train.base_lr`로 수정
- top-level `loss` → `train.loss`로 수정
- top-level `class_weights` → `train.class_weights`로 수정
- 기존 모델명 형식 `lite_race` 유지
- 과하게 긴 100epoch 대신 70~80epoch + patience로 정리
- V4가 epoch 2에서 최고점을 찍은 점을 보고 LR을 낮춤

## 서버 실행

```bash
cd /home/ubuntu/road-damage-segmentation-portfolio
unzip -o ~/LRS_V5_UPDATE_PATCH.zip -d .
bash scripts/run_v5_update_train.sh
```

## 먼저 돌릴 것

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v5_A_corrected_aug_stable.yaml
```

## 추가 선택

recall을 더 밀고 싶으면:

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v5_B_corrected_aug_recall.yaml
```

오탐이 너무 많으면:

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v5_C_corrected_aug_precision.yaml
```

GPU 메모리 터지면:

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v5_D_safe_oom.yaml
```

## 결과 회수

학습 후:

```bash
bash scripts/collect_v5_evidence.sh literace_v5_A_corrected_aug_stable_s42
```

그러면 repo 폴더에 `LRS_V5_UPDATE_EVIDENCE_...zip`가 생깁니다.
