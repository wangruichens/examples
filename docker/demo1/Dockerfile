FROM python:3.6-slim

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World
ENV http_proxy 127.0.0.1:1080
ENV https_proxy 127.0.0.1:1080

CMD ["python", "app.py"]