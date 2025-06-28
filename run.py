import click 
from pathlib import Path

from modules.encoder import encode_to_hls

@click.group()
def cli():
    pass

@cli.command("package")
@click.argument("path", type=click.Path(exists=True, path_type=Path, readable=True))
@click.option("--output", "-o", help="Output directory", type=click.Path(path_type=Path, writable=True))
def package(path, output):
    if output is None:
        output = path.with_suffix("")
    encode_to_hls(path, output)


if __name__ == "__main__":
    cli()