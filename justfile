default:
    @just --list

install:
    uv sync

test *ARGS:
    uv run pytest {{ARGS}}

fmt:
    uv run ruff format .
    uv run ruff check --fix .

lint:
    uv run ruff check .

typecheck:
    uv run pyright gridlife/

check: fmt lint typecheck test

demo *ARGS:
    uv run python -m gridlife.demo {{ARGS}}
