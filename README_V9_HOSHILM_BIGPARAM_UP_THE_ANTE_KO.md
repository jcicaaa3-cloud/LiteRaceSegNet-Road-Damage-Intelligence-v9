# LRS v9 HoshiLM Project QA Big-Parameter Up-The-Ante

이 패키지는 기존 v6 LiteRaceSegNet 코어를 건드리지 않고, `v8_hoshilm_submission/hoshilm_kr` 안의 HoshiLM Project QA 파트를 실제로 더 큰 파라미터 규모로 확장한 제출/시연용 버전입니다.

## 핵심 변경

기존 Project QA 기본 config는 다음 수준이었습니다.

- config: `configs/hoshilm_project_qa.yaml`
- tokenizer: char
- n_layer: 6
- n_head: 6
- n_embd: 384
- block_size: 384
- 실제 추정 파라미터: 약 11.03M

이번 Big-Parameter 기본 실행은 다음 config를 사용합니다.

- config: `configs/hoshilm_project_qa_xl.yaml`
- tokenizer: SentencePiece
- n_layer: 12
- n_head: 12
- n_embd: 768
- block_size: 512
- 실제 추정 파라미터: 약 85.96M

추가로 선택 실행용 200M급 실험 config도 포함했습니다.

- config: `configs/hoshilm_project_qa_xxl_200m_experimental.yaml`
- n_layer: 16
- n_head: 16
- n_embd: 1024
- block_size: 768
- 실제 추정 파라미터: 약 203.00M

## 실행

AWS에서 기본 Big-Parameter Project QA 학습:

```bash
cd ~/lrs_v9/road-damage-segmentation-portfolio/v8_hoshilm_submission/hoshilm_kr
bash run_project_qa_build.sh
bash run_project_qa_train.sh
bash run_project_qa_web.sh
```

`run_project_qa_train.sh`는 이제 기본적으로 `configs/hoshilm_project_qa_xl.yaml`을 사용합니다.

작은 버전으로 빠르게 확인만 할 때:

```bash
bash run_project_qa_train_small.sh
```

200M급 실험 버전을 돌릴 때:

```bash
bash run_project_qa_train_xxl_experimental.sh
```

## 주의

- HoshiLM은 LiteRaceSegNet segmentation 성능을 높이는 본체가 아닙니다.
- HoshiLM은 결과 로그, dataset verification, config, train metrics를 바탕으로 프로젝트 설명을 돕는 Project QA 보조 모듈입니다.
- 86M/203M으로 파라미터를 키워도 현재 corpus가 작기 때문에 상용 LLM급 성능을 주장하면 안 됩니다.
- 발표에서는 “prediction model은 LiteRaceSegNet, reporting/support interface는 HoshiLM Project QA”라고 선을 긋는 것이 안전합니다.
