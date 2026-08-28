ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip git libcurl4-openssl-dev libgomp1

COPY requirements.txt   requirements.txt
COPY requirements       requirements

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

WORKDIR /app

COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/* /app/ && \
    find build -name "*.so" -exec cp {} /app/ \;

# tools.sh invokes ./llama-* from /app, next to the shared libraries copied above
ENV LD_LIBRARY_PATH=/app
ENV LC_ALL=C.utf8

ENTRYPOINT ["/app/.devops/tools.sh"]
