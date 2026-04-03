default:
    @just --list

install:
    uv sync

serve *ARGS:
    .venv/bin/gridlife serve {{ARGS}}

test *ARGS:
    uv run pytest {{ARGS}}

testv *ARGS:
    uv run pytest -v {{ARGS}}

fmt:
    uv run ruff format .
    uv run ruff check --fix .

lint *ARGS:
    uv run ruff check . {{ARGS}}

typecheck:
    uv run pyright gridlife/

check: fmt lint typecheck test

demo *ARGS:
    .venv/bin/python -m gridlife.demo {{ARGS}}
