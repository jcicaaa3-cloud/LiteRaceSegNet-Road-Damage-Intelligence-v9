# AWS GPU Runbook — Quick Reference

## HoshiLM Project QA

```bash
cd ~/lrs_v9/road-damage-segmentation-portfolio/v8_hoshilm_submission/hoshilm_kr
bash run_project_qa_build.sh
bash run_project_qa_train.sh
bash run_project_qa_web.sh
```

웹 접속:

```text
http://AWS_PUBLIC_IP:8000
```

EC2 보안 그룹에서 TCP 8000 인바운드 허용이 필요합니다.

## GPU 확인

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## 결과 확인

```bash
ls -lah runs/hoshilm_project_qa
cat runs/hoshilm_project_qa/sample_output.txt
```
