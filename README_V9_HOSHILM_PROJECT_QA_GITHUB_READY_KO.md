# LiteRaceSegNet v9 — HoshiLM Project QA Extension

이 버전은 HoshiLM을 단순 장식용이 아니라 **결과 데이터 기반 Project QA**로 확장한 버전입니다.

## 핵심 변경

- `lrs_v6_evidence_summary`의 `train_log.csv`, dataset verify, leakage check, config YAML을 읽어 `project_facts.json` 생성
- 결과/설정/README 문서를 합쳐 `project_qa_corpus.txt` 생성
- HoshiLM 학습용 corpus를 `data/data_project_qa.txt`로 확장
- CLI 대화: `project_qa_chat.py`
- 웹 대화 UI: `web_project_qa/index.html`
- 로컬 API: `project_qa_api.py`
- HTML에 답변을 HTML에 직접 고정하지 않고, 브라우저가 `/api/chat`으로 질문을 보내면 Python QA 엔진이 답변

## AWS 실행 순서

```bash
cd ~/lrs_v9/road-damage-segmentation-portfolio/v8_hoshilm_submission/hoshilm_kr
bash run_project_qa_build.sh
bash run_project_qa_train.sh
bash run_project_qa_web.sh
```

브라우저 접속:

```text
http://AWS_PUBLIC_IP:8000
```

EC2 보안 그룹에서 8000번 포트 인바운드를 열어야 외부 브라우저에서 보입니다.

## 빠른 확인만 할 때

```bash
cd ~/lrs_v9/road-damage-segmentation-portfolio/v8_hoshilm_submission/hoshilm_kr
bash run_project_qa_smoke.sh
```

## 발표용 설명

HoshiLM은 대형 상용 LLM이 아니라 소형 decoder-only Transformer 언어모델 실험입니다. 이번 v9 Project QA 버전은 LiteRaceSegNet 결과 파일을 기반으로 QA corpus와 facts를 만들고, 이를 학습 및 검색 근거로 사용합니다. 웹 UI는 고정 답변 HTML이 아니라 로컬 Python API와 연결되어 질문마다 결과 데이터를 검색해 답합니다.

## 안전한 한계 설명

- 실사용 챗봇 성능을 주장하지 않음
- 결과 수치가 없는 질문에는 추측하지 않음
- evidence summary에 있는 수치만 답변 근거로 사용
- HoshiLM 생성문은 보조이며, 최종 근거는 `project_facts.json`과 corpus
