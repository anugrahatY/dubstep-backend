FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY backend/requirements.txt ./backend/requirements.txt
COPY backend/model-requirements.txt ./backend/model-requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
ARG INSTALL_LOCAL_MODEL_DEPS=false
RUN if [ "$INSTALL_LOCAL_MODEL_DEPS" = "true" ]; then pip install --no-cache-dir -r backend/model-requirements.txt; fi
COPY backend/ ./backend/
COPY modelbackend/ ./modelbackend/

RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)" || exit 1

CMD ["uvicorn", "--app-dir", "backend", "server:app", "--host", "0.0.0.0", "--port", "8000"]
