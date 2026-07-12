# Security Policy

## Reporting a vulnerability

If you discover a security issue in AutoShip CLI, report it privately to `team@autoship.dev`. Do not open a public issue.

You can expect:

- A response within 72 hours confirming receipt
- An assessment within 7 days, including a timeline for a fix
- CVE assignment if applicable
- Credit in the release notes (unless you prefer to remain anonymous)

## Scope

The following fall within scope:

- Code injection or command injection via crafted inputs (commit messages, config files, plugin manifests)
- Credential leakage — API keys, tokens, or environment variables exposed in logs, error messages, or artifacts
- Unsafe handling of user-supplied TOML, JSON, or other structured inputs
- Privilege escalation through the plugin system or LSP server
- Supply chain issues in bundled dependencies

Out of scope:

- Issues in third-party LLM providers (report those to the provider directly)
- Theoretical attacks requiring physical access or already-compromised machines
- Social engineering

## Supported versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | Yes       |
| 1.x     | No        |

## Dependencies

AutoShip uses Semgrep for code security scanning, pip-audit for dependency auditing, and cryptographic primitives from the `cryptography` library. Dependencies are pinned with hashes in `uv.lock`.

If you notice a vulnerable dependency in `uv.lock`, include the CVE ID and affected version range in your report.
