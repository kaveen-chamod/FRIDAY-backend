FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRIDAY_HOST=0.0.0.0 \
    FRIDAY_MEMORY_PATH=/var/lib/friday/memory/long_term.json \
    FRIDAY_STAGING_DIR=/var/lib/friday/staging_tools \
    HOME=/var/lib/friday

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system friday \
    && useradd --system --gid friday --home-dir /var/lib/friday friday \
    && mkdir -p /var/lib/friday /var/lib/friday/memory /var/lib/friday/staging_tools /app/config \
    && chown -R friday:friday /var/lib/friday /app

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY --chown=friday:friday . .
USER friday

EXPOSE 8000
VOLUME ["/var/lib/friday"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["python", "-m", "app.main"]
