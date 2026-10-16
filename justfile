set dotenv-load

test:
    uv run pytest

check:
    uv run ruff format --check .
    uv run ruff check .

fmt:
    uv run ruff format .
    uv run ruff check --fix .

logs:
    #!/usr/bin/env bash
    set -euo pipefail
    url=http://localhost:6006
    if ! curl -sf "$url" >/dev/null; then
        mkdir -p logs
        cmd="uv run tensorboard --logdir logs --port 6006"
        if [ -n "${TMUX:-}" ]; then
            tmux new-window -d -n tensorboard -c "$PWD" "$cmd"
        else
            tmux new-session -d -s tensorboard -c "$PWD" "$cmd"
        fi
        for _ in $(seq 120); do curl -sf "$url" >/dev/null && break; sleep 0.5; done
    fi
    if command -v open >/dev/null; then open "$url"; else xdg-open "$url"; fi

# vast.ai GPU instances: `just vast --help`, e.g. `just vast up`, `just vast train num_envs=16384`
vast *args:
    uv run python tools/vast.py {{args}}
