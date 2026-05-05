FROM python:3.11-slim

# OpenCV needs these system libraries even with the headless wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Preserve the repo layout in the image so app.services.scoring's
# parents[3]-based benchmarks path resolves correctly.
WORKDIR /app

COPY apps/api/requirements.txt ./apps/api/requirements.txt
RUN pip install --no-cache-dir -r ./apps/api/requirements.txt

COPY apps/api ./apps/api
COPY packages ./packages

WORKDIR /app/apps/api

ENV OUTPUT_DIR=/tmp/golf-analyzer-output
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
