# torch 2.7.1 + CUDA 12.8: required for Blackwell GPUs (sm_120), also works on Ampere/Ada/Hopper
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -c "import transformers, torch; print('BUILD CHECK:', transformers.__version__, torch.__version__); assert transformers.__version__.startswith('4.')"

# Bake weights + remote code at a pinned revision (no runtime downloads, no code drift)
RUN HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; \
snapshot_download('bharatgenai/AyurParam', revision='27c69eeef492bbc9e2c0048fc1e95959ffc0820e', local_dir='/models/AyurParam')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
