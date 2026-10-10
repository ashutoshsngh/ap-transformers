FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

# Install application dependencies without replacing PyTorch
COPY requirements.txt .
RUN python -m pip install --no-cache-dir \
    runpod transformers accelerate safetensors sentencepiece

# Verify PyTorch and CUDA availability during the build
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Copy the serverless handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
