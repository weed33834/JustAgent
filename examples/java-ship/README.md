# myagent-java-ship

A language-aware MyAgent-CLI plugin pack for Java projects. It hooks into
the `clean` and `verify` lifecycle to report Java-specific build artifacts
and suggest the conventional test command for the build tool in use
(`mvn test` or `gradle test`).

This is an **example / skeleton** pack. It is non-destructive: it only
observes and reports artifacts, it does not delete them.

## What it does

- `pre_clean`: reports Java build artifacts currently present
  (`target/classes`, `target/test-classes`, loose `*.class` files).
- `post_clean`: reports which of those artifacts were removed during the
  clean run.
- `pre_verify`: when a `pom.xml` or `build.gradle*` is present, suggests the
  build-tool-appropriate test command and `mvn checkstyle:check` as the
  linter.

## Build tool detection

`pre_verify` picks the test command based on the build file present:

| Build file                | Test command  |
|---------------------------|---------------|
| `pom.xml`                 | `mvn test`    |
| `build.gradle` / `.kts`   | `gradle test` |

## Artifact scope note

`target/classes` and `target/test-classes` are scoped explicitly to avoid
treating the whole `target/` tree (which also holds Maven metadata, reports,
and the local repository cache) as a single cleanable artifact. Loose
`*.class` files are matched at the project root only; extend the glob in
`src/myagent_java_ship/plugin.py` if you need to scan source directories.

## Install

```bash
pipx install myagent-java-ship
```

Or, for local development:

```bash
cd examples/java-ship
pip install -e .
```

The `[project.entry-points."myagent.plugins"]` table in `pyproject.toml`
registers the plugin automatically; no manual wiring is required.

## Config snippet

Add `mvn` and `gradle` to the verify allowlist (the default allowlist is
Python-centric):

```toml
# .myagent.toml
[verify]
allowed_commands = ["pytest", "mvn", "gradle"]
```

## Test

```bash
cd examples/java-ship
pytest
```

## Notes

- Loose `*.class` matching is root-level by default. For projects that emit
  class files into `src/`, adjust the glob in `plugin.py`.
- This pack is part of the 2.0.0 forward-expansion examples and is not
  distributed via the official plugin registry.
