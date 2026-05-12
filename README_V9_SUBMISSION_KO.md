# LRS v9 제출용 정리 노트

이 패키지는 제출용 정리본입니다.

## 유지한 것
- LiteRaceSegNet v6 기반 프로토타입/세그멘테이션 코드
- v8 HoshiLM-KR 언어모델 학습 코드
- HoshiLM-KR 학습 데이터: `v8_hoshilm_submission/hoshilm_kr/data/data.txt`
- tokenizer/config/model/train/generate 코드
- LiteRaceSegNet 핵심 evidence 및 정적 웹 쇼케이스

## 제거한 것
- 선택형 챗 설명 레이어
- 웹 데모 안의 사전 작성 Q&A / mock assistant UI
- evidence builder 안의 챗 예시 생성 경로

## HTML 관련 메모
`web_demo/index.html`은 언어모델 학습 데이터가 아니라, 제출/시연용 정적 웹 페이지입니다.
HoshiLM-KR의 실제 텍스트 학습 데이터는 `hoshilm_kr/data/data.txt`입니다.
검토 결과 해당 학습 데이터에는 `<!doctype html>` 또는 `<html>` 태그가 포함되어 있지 않습니다.

## 제출 프레이밍
HoshiLM-KR은 대형 상용 LLM 또는 배포형 챗봇이 아니라, decoder-only Transformer의 tokenization, causal self-attention, next-token prediction, checkpointing, sampling 흐름을 재현하는 소형 학습 실험 모듈로 설명하는 편이 안전합니다.
