# Asset and copyright policy

This repository is prepared for public portfolio use. Keep it clean before pushing to GitHub.

## Safe to commit

- source code written for this project
- configuration files
- small placeholder files such as `.gitkeep`
- documentation
- architecture diagrams authored for this project
- result templates without real private data

## Do not commit

- raw road images
- dataset masks or labels
- downloaded public datasets
- paid or restricted datasets
- private camera footage
- trained checkpoints
- pretrained model weights
- Hugging Face cache folders
- generated overlays if the input image license is unclear
- `.env`, API keys, cloud credentials
- thesis DOCX/PDF drafts containing personal information or school formatting

## Repository copyright scope

`LICENSE` and `NOTICE.txt` cover LiteRaceSegNet-related code, documentation, diagrams, experiment notes, configuration files, and scripts authored for this portfolio project.

This repository is not released as an MIT-licensed open-source project. It is public for portfolio viewing and academic demonstration only.

작성자의 사전 허가 없이 본 프로젝트 또는 그 구성 요소를 복제, 재배포, 수정, 공개, 2차 저작물 제작, 상업적 목적으로 사용하는 것을 허용하지 않습니다.

External third-party packages, model weights, datasets, APIs, and referenced model implementations are not included and are not relicensed.

## Dataset wording for README or resume

Use this wording:

> The repository does not include datasets or weights. Experiments require the user to place permitted image-mask pairs under the documented dataset layout.

Avoid this wording:

> Dataset is included.
> We provide pretrained weights.
> Anyone can freely use all assets in this repository.

## Result image policy

Only publish overlay images when the source road image can be redistributed. If the license is unclear, keep overlays local and publish only aggregate metrics.