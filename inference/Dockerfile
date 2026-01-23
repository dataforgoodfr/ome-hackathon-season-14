FROM python:3.12-slim-bookworm

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ common/
COPY inference/ inference/

ENTRYPOINT [ "python", "inference/predict.py" ]
