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
For DeepSeek, use `API_BASE_URL=https://api.deepseek.com`.
The app accepts either `HF_TOKEN` or `API_KEY` for the model credential. When using DeepSeek, set one of them to your DeepSeek API key.

```bash
export API_BASE_URL=https://api.deepseek.com
export MODEL_NAME=deepseek-chat
export API_KEY=your_deepseek_api_key

python inference.py
```

Without model credentials, the baseline falls back to a shortest-job-first heuristic so the script still completes for smoke testing.

## Using DeepSeek

1. Go to the DeepSeek platform and create an API key.
2. Set `API_BASE_URL` to `https://api.deepseek.com`.
3. Choose a model such as `deepseek-chat` or `deepseek-reasoner`.
4. Set `API_KEY` or `HF_TOKEN` to your DeepSeek API key.
5. Run `python inference.py`.

### Windows `cmd` setup

```cmd
set API_BASE_URL=https://api.deepseek.com
set MODEL_NAME=deepseek-chat
set API_KEY=your_deepseek_api_key
python inference.py
```

### Call DeepSeek API directly from `cmd`

```cmd
set API_KEY=your_deepseek_api_key
curl https://api.deepseek.com/chat/completions ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer %API_KEY%" ^
  -d "{\"model\":\"deepseek-chat\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a CPU scheduling assistant.\"},{\"role\":\"user\",\"content\":\"Given ready queue P1(8), P2(2), P3(1), which process should run next? Reply in one sentence.\"}]}"
```

This direct API call is useful for testing your DeepSeek key before running the full benchmark.
