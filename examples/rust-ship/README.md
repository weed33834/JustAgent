# myagent-rust-ship

A language-aware MyAgent-CLI plugin pack for Rust projects. It hooks into
the `clean` and `verify` lifecycle to report Rust-specific build artifacts
and suggest the conventional `cargo test` command.

This is an **example / skeleton** pack. It is non-destructive: it only
observes and reports artifacts, it does not delete them.

## What it does

- `pre_clean`: reports the per-profile build output directories present
  (`target/debug`, `target/release`).
- `post_clean`: reports which of those directories were removed during the
  clean run.
- `pre_verify`: when a `Cargo.toml` is present, suggests `cargo test` as the
  verification command and `cargo clippy` as the linter.

## Why not `target/` wholesale?

`target/` also holds cargo's shared dependency cache and incremental
compilation state. Wiping the whole directory is what `cargo clean` is for
and is intentionally **out of scope** for a `clean`-time hook. This pack only
matches the per-profile output directories (`target/debug`,
`target/release`), so it never treats cargo's cache as a cleanable artifact.

## Install

```bash
pipx install myagent-rust-ship
```

Or, for local development:

```bash
cd examples/rust-ship
pip install -e .
```

The `[project.entry-points."myagent.plugins"]` table in `pyproject.toml`
registers the plugin automatically; no manual wiring is required.

## Config snippet

Add `cargo` to the verify allowlist so `myagent verify "cargo test"` is
permitted (the default allowlist is Python-centric):

```toml
# .myagent.toml
[verify]
allowed_commands = ["pytest", "cargo"]
```

## Test

```bash
cd examples/rust-ship
pytest
```

## Notes

- Only `target/debug` and `target/release` are recognised; other profile
  outputs (e.g. `target/x86_64-unknown-linux-gnu/...`) can be added to
  `ARTIFACTS` in `src/myagent_rust_ship/plugin.py` if needed.
- This pack is part of the 1.1.0 forward-expansion examples and is not
  distributed via the official plugin registry.
