FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    gcc patchelf clang ccache

RUN pip install Nuitka

RUN pip install --no-cache-dir -r requirements.txt
ENV CC=clang

RUN python -m nuitka --onefile \
    --module-parameter=torch-disable-jit=no \
    run.py

RUN ls -la

CMD [ "./run.bin" ]