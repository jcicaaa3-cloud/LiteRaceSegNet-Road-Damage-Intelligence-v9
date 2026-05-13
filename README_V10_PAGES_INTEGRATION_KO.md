# LiteRaceSegNet V10 Pages Integration

V10은 v9 저장소에 GitHub Pages용 랜딩 페이지와 개선된 README 이미지를 추가한 정리본입니다.

## 추가/변경된 항목

- `index.html`
  - GitHub Pages 루트 랜딩 페이지
  - v7 Workstation Demo 링크 통합
  - v9/v10 Static Prototype 링크
  - HoshiLM Project QA UI 링크

- `.nojekyll`
  - GitHub Pages에서 정적 파일 경로 처리를 단순화하기 위한 파일

- `docs/assets/literace_architecture_clean.png`
  - 기존 러프 구조도를 GitHub README용 고품질 이미지로 교체

- `docs/github_assets/project_qa_flow.png`
  - Project QA 흐름도를 고품질 이미지로 교체

- `docs/github_assets/repo_structure.png`
  - 저장소 구조 이미지를 고품질 이미지로 교체

## GitHub Pages 설정

GitHub 저장소에서 다음 순서로 설정합니다.

```text
Settings
→ Pages
→ Build and deployment
→ Source: Deploy from a branch
→ Branch: main
→ Folder: / root
→ Save
```

Pages 주소 예상:

```text
https://jcicaaa3-cloud.github.io/LiteRaceSegNet-Road-Damage-Intelligence-v9/
```

## 주의

`v8_hoshilm_submission/web_project_qa/`는 GitHub Pages에서 UI 미리보기는 가능하지만, 실제 답변은 로컬 또는 AWS에서 Python API 서버를 실행해야 합니다.

```bash
cd v8_hoshilm_submission/hoshilm_kr
bash run_project_qa_build.sh
bash run_project_qa_train.sh
bash run_project_qa_web.sh
```

