# GitHub Upload Runbook — LiteRaceSegNet v9

![GitHub upload flow](github_assets/github_upload_flow.png)

## 0. 업로드 전 확인

GitHub에는 학습 결과물 전체, 개인키, `.env`, AWS `.pem`, 대형 checkpoint를 올리지 않는 것을 권장합니다.
이 패키지는 `.gitignore`에 `runs/`, `*.pem`, `.env`, `*.pt`, `*.pth` 등을 제외하도록 정리했습니다.

```bash
git status --ignored
find . -type f -size +50M -print
```

## 1. 새 GitHub 저장소 만들기

GitHub에서 새 repository를 만들고, README / .gitignore / license는 일단 생성하지 않는 편이 깔끔합니다. 이미 이 패키지 안에 README와 .gitignore가 들어 있습니다.

권장 저장소 설명:

```text
Lightweight road-damage segmentation with LiteRaceSegNet, SegFormer baseline comparison, and optional HoshiLM Project QA interface.
```

## 2. PowerShell 업로드 명령어

아래에서 `YOUR_GITHUB_ID`와 `YOUR_REPO_NAME`만 바꾸면 됩니다.

```powershell
cd "C:\Users\본인\Downloads\road-damage-segmentation-portfolio"

git init
git branch -M main
git add .
git commit -m "docs: publish LiteRaceSegNet v9 GitHub package"
git remote add origin https://github.com/YOUR_GITHUB_ID/YOUR_REPO_NAME.git
git push -u origin main
```

## 3. 이미 git 저장소였던 경우

```powershell
git remote -v
git status
git add .
git commit -m "docs: update GitHub-ready v9 package"
git push origin main
```

## 4. 브라우저 업로드 방식

작은 파일만 추가할 때는 GitHub 웹에서 `Add file` → `Upload files`를 사용할 수 있습니다. 다만 `.gitattributes` 같은 Git 설정이 필요한 경우에는 웹 업로드보다 Git push가 안전합니다.

## 5. 업로드 후 확인

GitHub repository 첫 화면에서 아래를 확인합니다.

- README 이미지가 깨지지 않는지
- `docs/assets/` 이미지가 보이는지
- `v8_hoshilm_submission/hoshilm_kr/`가 들어갔는지
- `runs/`, `.pem`, `.env`, 대형 checkpoint가 올라가지 않았는지

```bash
git ls-files | grep -E "(\.pem|\.env|runs/|best\.pt|last\.pt|best\.pth|last\.pth)"
```

결과가 비어 있으면 안전합니다.

## 6. GitHub Pages 주의

`web_demo/index.html`처럼 완전 정적 HTML은 GitHub Pages에 올릴 수 있습니다.  
반면 `web_project_qa/index.html`은 `/api/chat` Python API가 필요하므로 GitHub Pages 단독으로는 동작하지 않습니다. 이 경우 AWS나 로컬에서 `run_project_qa_web.sh`로 실행해야 합니다.
