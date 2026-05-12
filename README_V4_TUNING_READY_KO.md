# LRS AWS 반환물 v4 튜닝 준비본

이 ZIP은 기존 AWS 반환물에 `final_evidence/07_v4_tuning_patch`와 `seg/config`, `scripts`를 추가한 버전이다.

중요:
- 기존 결과값 CSV를 조작하지 않았다.
- 새로 추가한 것은 실제 재학습을 위한 하이퍼파라미터 설정과 실행 스크립트다.
- AWS repo 루트에 이 ZIP을 풀고 `scripts/run_v4_train_sweep.sh`를 실행하면 된다.

추천:
1. `v4_A_highres_damage_boost` 먼저 실행
2. OOM이면 `v4_C_safe_no_oom`
3. recall을 더 밀고 싶으면 `v4_B_recall_aggressive`
4. 오탐이 너무 많으면 `v4_D_precision_recover`
