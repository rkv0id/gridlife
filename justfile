default:
    @just --list

install:
    uv sync

# Local mode (default, no Ray)
serve *ARGS:
    .venv/bin/gridlife serve {{ARGS}}

# Local Ray mode (for testing Ray behavior)
serve-ray *ARGS:
    .venv/bin/gridlife serve --local-ray {{ARGS}}

# Run headless simulation
run *ARGS:
    .venv/bin/gridlife run {{ARGS}}

# List available simulations
list:
    .venv/bin/gridlife list

test *ARGS:
    uv run pytest {{ARGS}}

testv *ARGS:
    uv run pytest -v {{ARGS}}

# Run Ray integration tests
test-ray:
    .venv/bin/pytest -m ray -v

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

# Generate demo outputs
demo-png:
    .venv/bin/gridlife run --sim game_of_life --width 512 --height 512 --steps 100 --output gol.png
    .venv/bin/gridlife run --sim gray_scott --width 512 --height 512 --steps 500 --preset mitosis --output gs.png
    .venv/bin/gridlife run --sim lenia --width 512 --height 512 --steps 200 --output lenia.png
    .venv/bin/gridlife run --sim smoothlife --width 512 --height 512 --steps 200 --output smoothlife.png

demo-gif:
    .venv/bin/gridlife run --sim gray_scott --width 512 --height 512 --steps 2000 --preset coral --output demo.gif --fps 15
