FROM python:3.8-slim-buster

RUN python -m pip install jupyter

WORKDIR /app
COPY . /app
RUN python -m pip install .[examples]

WORKDIR /app/examples

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
