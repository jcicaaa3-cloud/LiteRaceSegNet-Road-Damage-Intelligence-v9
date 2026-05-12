# LRS V6 Visual Evidence Hotfix

목적: V6 학습 결과 ZIP 안에 숫자만 있고 overlay/service card 이미지가 부족한 문제를 보완한다.

추가되는 파일:
- `scripts/make_v6_visual_evidence.py`: best/last checkpoint로 val 10장 prediction overlay 생성
- `scripts/collect_v6_evidence_with_visuals.sh`: 숫자, 누수 체크, pair 검증, overlay/service card까지 포함한 evidence ZIP 생성

실행 위치:
`/home/ubuntu/road-damage-segmentation-portfolio`

실행:
```bash
cd /home/ubuntu/road-damage-segmentation-portfolio
source .venv/bin/activate
bash scripts/collect_v6_evidence_with_visuals.sh literace_v6_A_newval_balanced_s42
```

출력:
`LRS_V6_NEWVAL_EVIDENCE_WITH_VISUALS_literace_v6_A_newval_balanced_s42.zip`

정상 체크:
- `oldval_pothole in train should be 10`
- `oldval_newval in train should be 0`
- `newval in train should be 0`
- `newval in val should be 10`
- `datasets/pothole_binary_aug_v6/processed` train images/masks가 2400/2400 근처

주의:
- 학습 중에도 업로드는 가능하지만, evidence 생성은 학습이 끝난 뒤 실행하는 것이 가장 안전하다.
- 이 스크립트가 만드는 visual metric은 시각화 sanity check다. 공식 수치는 `train_log.csv`와 `summary_best_last.json` 기준으로 사용한다.


## V3_SAFE 변경점
- `--norm auto` 제거: validation GT mask를 보고 raw/imagenet 중 더 좋은 전처리를 고르지 않는다.
- 기본값은 `--norm config`이며, config에서 ImageNet normalization 흔적이 없으면 raw `[0,1]` 전처리를 사용한다.
- Prediction mask는 `model(image)`에서 생성되며, GT mask는 metric 계산과 GT overlay 저장에만 사용한다.
- 이 버전의 visual metric은 validation-label tuning 없이 생성된 sanity check로 쓰는 것이 목적이다.


## V4_FIXED_IMAGENET 변경점
- validation GT를 보고 전처리를 고르지 않는다.
- visual evidence 생성 시 `--norm imagenet`을 명시적으로 고정한다.
- 목적: 학습/로그에서 사용된 ImageNet normalization 계열 전처리와 inference visual evidence를 맞추기 위한 버전이다.
- V2의 `auto`처럼 점수 좋은 전처리를 고르는 방식이 아니다.
