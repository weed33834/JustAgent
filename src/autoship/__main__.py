"""Entry point for `python -m autoship`."""

from autoship.cli.main import cli_entrypoint

if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
