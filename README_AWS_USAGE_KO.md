# LRS v9 AWS 사용법

## 1) 압축 해제 후 위치

```bash
mkdir -p ~/lrs_v9
cd ~/lrs_v9
unzip ~/LRS_v9_submission_no_llmservice_AWS_READY.zip
cd road-damage-segmentation-portfolio
```

## 2) HoshiLM 빠른 실행 확인

```bash
cd ~/lrs_v9/road-damage-segmentation-portfolio/v8_hoshilm_submission/hoshilm_kr
chmod +x run_aws_smoke.sh run_aws.sh
bash run_aws_smoke.sh
```

결과물은 아래 경로에 생성된다.

```text
v8_hoshilm_submission/hoshilm_kr/runs/hoshilm_smoke_aws/best.pt
v8_hoshilm_submission/hoshilm_kr/runs/hoshilm_smoke_aws/last.pt
v8_hoshilm_submission/hoshilm_kr/runs/hoshilm_smoke_aws/train_log.csv
v8_hoshilm_submission/hoshilm_kr/runs/hoshilm_smoke_aws/sample_output.txt
```

## 3) HoshiLM 본 실행

```bash
cd ~/lrs_v9/road-damage-segmentation-portfolio/v8_hoshilm_submission/hoshilm_kr
bash run_aws.sh
```

`run_aws.sh`는 `configs/hoshilm_m.yaml` 기준으로 실행한다.

## 4) 웹 데모 확인

웹 데모는 정적 HTML이다. 서버 실행 없이 파일만 열어도 된다.

```text
v8_hoshilm_submission/web_demo/index.html
```

AWS에서 임시로 확인하려면 다음처럼 실행할 수 있다.

```bash
cd ~/lrs_v9/road-damage-segmentation-portfolio/v8_hoshilm_submission/web_demo
python3 -m http.server 8000
```

브라우저에서 `http://AWS_PUBLIC_IP:8000` 접속. 단, EC2 보안 그룹에서 8000번 포트를 열어야 한다.

## 5) 주의

- `llm_service/`와 `run_LLM_CHAT_SERVICE.bat`는 제출용 구조를 명확히 하기 위해 제거했다.
- 포함된 HoshiLM 데이터는 작은 smoke-test/재현용 텍스트라서 상용 LLM 품질을 기대하는 용도가 아니다.
- 학습 데이터와 v6 evidence 요약은 유지했다.
