# Environment setup

## Requirements

- Python 3.12 (verified on 3.13)
- No API key is needed for 388 of the 428 tests

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

`.env` is ignored by Git. Nothing in the book requires a key until
Chapter 5, and the default `LLMAPP_PROVIDER=fake` runs the whole
offline test tier.

## Verify

```bash
python scripts/verify_setup.py
```

A healthy run prints `Environment ready.` and exits zero. A
misconfigured one names the variable that is missing and exits one.

## Tokenizer cache

`tiktoken` downloads its encoding file on first use. To make the
test tier fully hermetic, pre-warm it:

```bash
export TIKTOKEN_CACHE_DIR=$PWD/.tiktoken
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

The Dockerfile does this at build time.
