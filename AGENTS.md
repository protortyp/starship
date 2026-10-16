# Agents

- Branches: `ce/<feature-name>`, off `main`.
- Commit small and often; `just check` and `just test` must pass first.
- PRs: squash-merge only; title `feat(scope): ...` / `fix(scope): ...` / `chore(scope): ...`; body is one short paragraph.
- Merge once CI is green.
- GPU runs: `.env` holds `VAST_API_KEY`; `just vast up` rents and sets up an RTX 3090 (`--gpu` to change), `just vast train <hydra overrides>`, `just vast status`, `just vast logs` (TensorBoard via SSH tunnel), `just vast pull`, `just vast down` (see `just vast --help`).
