FROM python:3.12-bookworm

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY final/ .


ENTRYPOINT [ "python", "predict.py" ]
