# Sử dụng phiên bản Airflow 2.10.3 mới hơn để giải quyết xung đột thư viện
FROM apache/airflow:2.10.3-python3.10

USER root
# Cài đặt công cụ OCR cục bộ cho Parser
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow
# Cài đặt toàn bộ thư viện RAG
COPY requirements.txt /requirements.txt

# Nâng cấp pip lên bản mới nhất
RUN pip install --upgrade pip


RUN --mount=type=cache,target=/home/airflow/.cache/pip \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /requirements.txt