FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir snowballstemmer fastembed numpy

COPY kb.sqlite server.py search.py index.html pdf_map.json /app/
COPY pdf /app/pdf

# Ключ доступа задаётся переменной окружения RAG_ACCESS_KEY в настройках хостинга.
# Порт предоставляет платформа через переменную PORT; по умолчанию 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "python server.py --port ${PORT} --no-browser"]
