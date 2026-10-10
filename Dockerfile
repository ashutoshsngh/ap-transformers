FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch && \
    python -m pip install --no-cache-dir runpod transformers accelerate safetensors sentencepiece

COPY handler.py .

RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

CMD ["python", "-u", "handler.py"]
