# CPU image: data preparation, reference-backend evaluation, analysis and the
# full test suite. For GPU training, use the CUDA stage below or install
# ".[train]" on a CUDA host with a matching driver.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rineya-lab

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[validate,symbolic,dev]"

COPY configs ./configs
COPY schemas ./schemas
COPY scripts ./scripts
COPY docs ./docs
COPY tests ./tests

# Mount the dataset read-only at /data:
#   docker run -v /host/rineya_sequence_independence_dataset_v1:/data:ro ...
ENV RINEYA_DATASET_ROOT=/data
VOLUME ["/data", "/opt/rineya-lab/runs"]

RUN useradd --create-home --uid 1000 lab && chown -R lab:lab /opt/rineya-lab
USER lab

HEALTHCHECK --interval=1m --timeout=30s --retries=2 \
    CMD python -c "import rineya_lab; print(rineya_lab.__version__)" || exit 1

ENTRYPOINT ["rineya-lab"]
CMD ["--help"]


# ---------------------------------------------------------------------------
# GPU training stage. Build with:
#   docker build --target gpu -t rineya-lab:gpu .
# Match the CUDA tag to the host driver.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /opt/rineya-lab
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[train,quant,validate,symbolic]"

COPY configs ./configs
COPY schemas ./schemas
COPY scripts ./scripts
COPY docs ./docs

ENV RINEYA_DATASET_ROOT=/data
VOLUME ["/data", "/opt/rineya-lab/runs"]
ENTRYPOINT ["rineya-lab"]
CMD ["--help"]
