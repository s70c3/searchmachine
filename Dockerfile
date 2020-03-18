FROM python:3.8.2-slim-buster

RUN pip install six numpy catboost Flask

RUN mkdir /opt/service
WORKDIR /opt/service

COPY server.py server.py
COPY weights.cbm weights.cbm

EXPOSE 5000

CMD python server.py
