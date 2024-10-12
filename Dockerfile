FROM jupyter/minimal-notebook:python-3.10

ENV PYTHONUNBUFFERED=1

COPY ./project .

RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ServerApp.ip=0.0.0.0", "--ServerApp.port=8888", "--ServerApp.allow_root=True", "--ServerApp.token=''", "--ServerApp.password=''"]