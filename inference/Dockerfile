FROM python:3.12-bookworm

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY inference/logs.py .
COPY inference/models.py .
COPY inference/database_connection.py .
COPY inference/predict.py .

ENTRYPOINT [ "python", "predict.py" ]