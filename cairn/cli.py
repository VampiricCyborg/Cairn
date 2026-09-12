"""Typer entrypoint for the `cairn` CLI."""

import typer

from cairn import __version__

app = typer.Typer(help="Cairn: harness-agnostic, git-native memory for coding agents.")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show the Cairn version and exit."),
) -> None:
    if version:
        typer.echo(f"cairn {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
