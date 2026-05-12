# LiteRaceSegNet V6 New-Val Patch

목표는 유지한다.

> 도로 손상 영역을 정확하게 분할하는 경량 세그멘테이션 모델 LiteRaceSegNet 개선

이번 V6는 기존 목표를 낮추는 게 아니라, 새 validation 10쌍을 기준으로 다시 일반화 성능을 올리는 패치다.

## 이번 패치가 하는 일

1. 현재 `datasets/pothole_binary/processed/val` 10쌍을 `train`으로 편입한다.
2. 이때 파일명 충돌을 막기 위해 기존 val은 `oldval_` prefix를 붙여 train에 넣는다.
3. 사용자가 새로 만든 `LRS_NEW_VAL_10_PAIRS.zip`을 새 validation set으로 교체한다.
4. `datasets/pothole_binary/processed` 전체를 자동 백업한다.
5. V6용 증강 데이터셋을 `datasets/pothole_binary_aug_v6/processed`에 새로 만든다.
6. V6-A balanced config로 학습한다.
7. best/last checkpoint, train_log, config, manifest를 evidence ZIP으로 회수한다.

## 새 val ZIP 구조

파일명은 image/mask의 stem이 같아야 한다.

```text
LRS_NEW_VAL_10_PAIRS.zip
└─ new_val/
   ├─ images/
   │  ├─ pothole_new_001.jpg
   │  └─ ...
   └─ masks/
      ├─ pothole_new_001.png
      └─ ...
```

아래 구조도 자동 인식한다.

```text
LRS_NEW_VAL_10_PAIRS.zip
├─ images/
└─ masks/
```

## 실행 전 예상 상태

현재 서버의 원본 데이터셋이 아래처럼 정상이어야 한다.

```text
train images=110 masks=110
val images=10 masks=10
```

V6 준비 후 예상 상태:

```text
train images=120 masks=120
val images=10 masks=10
```

V6 증강 후 예상 상태는 대략 다음과 같다.

```text
train images≈2400 masks≈2400
val images=10 masks=10
```

V6 기본 증강은 원본 train 1쌍당 최대 20쌍을 만든다. mask가 비어 있는 원본이 있으면 train 증강 수는 조금 달라질 수 있다.

## 서버 실행

V6 패치와 새 val ZIP을 `/home/ubuntu`에 올린 뒤:

```bash
cd /home/ubuntu/road-damage-segmentation-portfolio
unzip -o ~/LRS_V6_PATCH_READY.zip -d .
bash scripts/run_v6_prepare_and_train.sh ~/LRS_NEW_VAL_10_PAIRS.zip
```

끊김 방지용 tmux:

```bash
tmux new -s lrs_v6
cd /home/ubuntu/road-damage-segmentation-portfolio
unzip -o ~/LRS_V6_PATCH_READY.zip -d .
bash scripts/run_v6_prepare_and_train.sh ~/LRS_NEW_VAL_10_PAIRS.zip
```

빠져나오기:

```text
Ctrl + b 누르고 d
```

다시 들어가기:

```bash
tmux attach -t lrs_v6
```

## 결과 회수

학습이 끝나면 자동으로 evidence ZIP이 생긴다.

```text
LRS_V6_NEWVAL_EVIDENCE_literace_v6_A_newval_balanced_s42.zip
```

수동 회수:

```bash
cd /home/ubuntu/road-damage-segmentation-portfolio
bash scripts/collect_v6_evidence.sh literace_v6_A_newval_balanced_s42
```

## 중요한 방어 장치

- 기존 val을 train에 넣을 때 `oldval_` prefix를 붙인다. `pothole_001` 같은 기존 train 파일을 덮어쓰지 않는다.
- 실행 전 전체 dataset을 `datasets/_v6_backup_날짜/pothole_binary_processed`에 백업한다.
- `oldval_` 파일이 이미 train에 있으면 중복 편입을 막기 위해 멈춘다.
- 새 val이 정확히 10쌍이 아니면 멈춘다.
- image/mask 크기가 다르면 멈춘다.
- mask가 비어 있으면 기본적으로 멈춘다.

## 선택 실험

V6-A가 기본이다.

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v6_A_newval_balanced.yaml
```

손상 recall을 더 밀고 싶으면:

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v6_B_newval_recall.yaml
```

오탐이 너무 크면:

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v6_C_newval_precision.yaml
```

CUDA OOM이면:

```bash
python seg/train_literace.py --config seg/config/pothole_binary_literace_v6_D_safe_oom.yaml
```
