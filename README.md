# Backend API

The API translates Japanese phrases into up to ten ranked Hindi adaptations using Groq. The key is read server-side from `backend/.env` or `GROQ_API_KEY`; it is never sent to the browser.

For the combined Docker deployment, follow the repository [deployment guide](../README.md). For local API development, install `backend/requirements.txt`, then run from the repository root:

```bash
uvicorn --app-dir backend server:app --host 127.0.0.1 --port 8000
```

The translator uses `openai/gpt-oss-120b` by default. Set `GROQ_MODEL` to choose another Groq model. The local adapter remains optional; when its adapter files exist in `backend/model/`, the existing `/api/v1/adapt` route is also enabled.
