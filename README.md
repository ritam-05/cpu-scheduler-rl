---
title: CPU Scheduler RL
sdk: docker
app_port: 7860
license: mit
pinned: false
---

# CPU Scheduler RL

An OpenEnv-style CPU scheduling benchmark with:

- a deterministic scheduling environment
- three graded tasks
- a baseline `inference.py` runner using an OpenAI-compatible client
- a FastAPI service for Hugging Face Spaces deployment

## Endpoints

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /grade/{task_id}`
- `GET /validate`

## Local Run

```bash
docker build -t cpu-scheduler-rl .
docker run -p 7860:7860 cpu-scheduler-rl
```

Or without Docker:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Baseline Inference

Defaults are set only for `API_BASE_URL` and `MODEL_NAME`.
`HF_TOKEN` has no default and must be provided when using a remote model.

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token_here

python inference.py
```

Without model credentials, the baseline falls back to a shortest-job-first heuristic so the script still completes for smoke testing.
