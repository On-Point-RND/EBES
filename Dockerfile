FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN apt-get update \
    && apt-get install -y openjdk-21-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install rsync

ADD requirements* /workspace

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

RUN mkdir -p /.local && chmod -R 777 /.local

ENV _JAVA_OPTIONS="-Xmx32g"

ARG DEV
RUN if [[ -n "$DEV" ]]; then \
        pip install --no-cache-dir -U pip \
        && pip install --no-cache-dir -r requirements-dev.txt \
        && rm requirements-dev.txt; \
    fi

WORKDIR /home

