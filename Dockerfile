# Build stage: compile wheels once, with the toolchain.
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Dependencies change less often than source, so they are
# copied and installed first. Editing a .py file then reuses
# this layer instead of reinstalling everything.
COPY requirements.txt ./
RUN pip wheel --wheel-dir /wheels -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels --no-deps .

# Runtime stage: no compiler, no build tools, no source.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

RUN useradd --create-home --uid 10001 llmapp

WORKDIR /app

COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels llmapp \
    && rm -rf /wheels

# Prompts and evaluation data are read at runtime.
COPY --chown=llmapp:llmapp prompts ./prompts
COPY --chown=llmapp:llmapp evals ./evals

# Warm the tokenizer cache so the container needs no network
# on its first request.
RUN mkdir -p /opt/tiktoken \
    && python -c "import tiktoken; \
tiktoken.get_encoding('o200k_base')" \
    && chown -R llmapp:llmapp /opt/tiktoken

USER llmapp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen( \
'http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["llmapp-serve"]
