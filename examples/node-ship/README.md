# autoship-node-ship

A language-aware AutoShip-CLI plugin pack for Node projects. It hooks into
the `clean` and `verify` lifecycle to report Node-specific build artifacts
and suggest the conventional test command for the package manager in use.

This is an **example / skeleton** pack. It is non-destructive: it only
observes and reports artifacts, it does not delete them.

## What it does

- `pre_clean`: reports Node build artifacts currently present at the project
  root (`dist/`, `build/`, `*.tsbuildinfo`, `coverage/`).
- `post_clean`: reports which of those artifacts were removed during the
  clean run.
- `pre_verify`: when a `package.json` is present, suggests the test command
  matching the detected package manager (`npm test`, `pnpm test`, or
  `yarn test`) and `npm run lint` as the linter.

## Package manager detection

`pre_verify` picks the test command based on the lockfile present:

| Lockfile           | Test command  |
|--------------------|---------------|
| `pnpm-lock.yaml`   | `pnpm test`   |
| `yarn.lock`        | `yarn test`   |
| (none / other)     | `npm test`    |

## Install

```bash
pipx install autoship-node-ship
```

Or, for local development:

```bash
cd examples/node-ship
pip install -e .
```

The `[project.entry-points."autoship.plugins"]` table in `pyproject.toml`
registers the plugin automatically; no manual wiring is required.

## Config snippet

`npm`, `yarn`, and `pnpm` are already in the default verify allowlist, so no
extra configuration is required. If you lock the allowlist down, make sure to
keep your package manager on it:

```toml
# .autoship.toml
[verify]
allowed_commands = ["pytest", "npm", "pnpm", "yarn"]
```

## Test

```bash
cd examples/node-ship
pytest
```

## Notes

- Artifacts are matched at the project root. For monorepos, extend
  `ARTIFACTS` in `src/autoship_node_ship/plugin.py` to walk workspaces.
- This pack is part of the 1.1.0 forward-expansion examples and is not
  distributed via the official plugin registry.
