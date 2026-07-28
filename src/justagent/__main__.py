"""Entry point for `python -m justagent`."""

from justagent.cli.main import cli_entrypoint

if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
