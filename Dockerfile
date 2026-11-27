# Container recipe used to deploy the app (e.g. on Hugging Face Spaces).
# Build:  docker build -t rag-chatbot .
# Run:    docker run -p 7860:7860 --env-file .env rag-chatbot   → http://localhost:7860

FROM python:3.13-slim

# Tesseract = the OCR program that reads text from scanned PDFs and images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as user 1000, so the app runs as that user here too.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Download the default embedding and re-ranking models now, so the app starts fast.
RUN python -c "from app.services.onnx_models import OnnxSentenceEncoder, OnnxCrossEncoder; \
OnnxSentenceEncoder('sentence-transformers/all-MiniLM-L6-v2'); \
OnnxCrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
