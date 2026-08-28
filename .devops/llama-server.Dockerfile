ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && \
    apt-get install -y build-essential cmake git libcurl4-openssl-dev curl

WORKDIR /app

COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON && \
    cmake --build build --config Release --target llama-server -j$(nproc)

FROM ubuntu:$UBUNTU_VERSION AS runtime

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libgomp1 curl

COPY --from=build /app/build/bin/llama-server           /llama-server
COPY --from=build /app/build/examples/mtmd/libmtmd.so   /usr/local/lib/
COPY --from=build /app/build/ggml/src/libggml.so        /usr/local/lib/
COPY --from=build /app/build/src/libllama.so            /usr/local/lib/
RUN ldconfig

ENV LC_ALL=C.utf8
# Must be set to 0.0.0.0 so it can listen to requests from host machine
ENV LLAMA_ARG_HOST=0.0.0.0

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/llama-server" ]
