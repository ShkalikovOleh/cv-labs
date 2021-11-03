FROM python:3.8-slim-buster

RUN python -m pip install jupyter 
RUN python -m pip install jax[cpu]==0.2.21 
RUN python -m pip install scikit-learn>=0.20 matplotlib>=2.2.0

WORKDIR /app
COPY . /app
RUN python -m pip install .[examples]

WORKDIR /app/examples

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
