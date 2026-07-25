# syntax=docker/dockerfile:1

FROM rust:1.93-slim-bookworm AS rust-toolchain

FROM python:3.12-slim-bookworm AS wheel-builder

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:${PATH}

COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup

# Debian's build-essential meta-package is intentionally sourced from the
# pinned bookworm base image snapshot used by this build.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir maturin==1.14.1

WORKDIR /src
COPY scheduler ./scheduler
COPY core ./core
COPY serve ./serve

RUN maturin build \
        --release \
        --features python \
        --manifest-path scheduler/Cargo.toml \
        --out /wheels \
    && python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels ./core ./serve

FROM python:3.12-slim-bookworm AS runtime

ENV HF_HOME=/home/autotree/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=wheel-builder /wheels /wheels

RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 \
    && python -m pip install --no-cache-dir \
        /wheels/autotree_scheduler-*.whl \
        /wheels/autotree_core-*.whl \
        /wheels/autotree_serve-*.whl \
        transformers==5.14.1 \
        safetensors==0.8.0 \
    && rm -rf /wheels \
    && useradd --create-home --uid 10001 autotree \
    && mkdir -p "${HF_HOME}" \
    && chown -R autotree:autotree /home/autotree

USER autotree
WORKDIR /home/autotree

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=4).read()"]

ENTRYPOINT ["autotree", "serve"]
CMD ["--engine", "treekv", "--model", "gpt2", "--host", "0.0.0.0", "--port", "8000"]
