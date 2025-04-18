FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN pip install --no-cache-dir matplotlib
WORKDIR /app
COPY . /app
CMD ["python", "main.py"]
