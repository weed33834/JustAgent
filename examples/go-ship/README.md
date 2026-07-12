# autoship-go-ship

A language-aware AutoShip-CLI plugin pack for Go projects. It hooks into the
`clean` and `verify` lifecycle to report Go-specific build artifacts and
suggest the conventional Go test command.

This is an **example / skeleton** pack. It is non-destructive: it only
observes and reports artifacts, it does not delete them. Extend it to suit
your own cleanup workflow.

## What it does

- `pre_clean`: reports Go build artifacts currently present at the project
  root (`bin/`, `*.test`, `*.out`).
- `post_clean`: reports which of those artifacts were removed during the
  clean run (the built-in formatter does not remove build outputs, so this
  is usually a no-op summary).
- `pre_verify`: when a `go.mod` is present, suggests `go test ./...` as the
  verification command and `golangci-lint run` as the linter.

## Install

```bash
pipx install autoship-go-ship
```

Or, for local development:

```bash
cd examples/go-ship
pip install -e .
```

The `[project.entry-points."autoship.plugins"]` table in `pyproject.toml`
registers the plugin automatically; no manual wiring is required.

## Config snippet

Add `go` to the verify allowlist so `autoship verify "go test ./..."` is
permitted (the default allowlist is Python-centric):

```toml
# .autoship.toml
[verify]
allowed_commands = ["pytest", "go"]
```

## Test

```bash
cd examples/go-ship
pytest
```

## Notes

- Artifacts are matched at the project root. Adjust `ARTIFACTS` in
  `src/autoship_go_ship/plugin.py` to scan subdirectories if needed.
- This pack is part of the 1.1.0 forward-expansion examples and is not
  distributed via the official plugin registry.
