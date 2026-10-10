FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
