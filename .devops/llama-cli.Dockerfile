ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && \
    apt-get install -y build-essential cmake git

WORKDIR /app

COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release --target llama-cli -j$(nproc)

FROM ubuntu:$UBUNTU_VERSION AS runtime

RUN apt-get update && \
    apt-get install -y libgomp1

COPY --from=build /app/build/bin/llama-cli        /llama-cli
COPY --from=build /app/build/ggml/src/libggml.so  /usr/local/lib/
COPY --from=build /app/build/src/libllama.so      /usr/local/lib/
RUN ldconfig

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/llama-cli" ]
