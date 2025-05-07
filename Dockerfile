FROM python:3.11-slim

# системные зависимости unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential poppler-utils tesseract-ocr libmagic-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
     && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
