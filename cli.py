import click
from main import CloudflarePages
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@click.group()
def cli():
    pass

@cli.command("upload")
@click.option("--token", help="Cloudflare API token", envvar="CLOUDFLARE_API_TOKEN")
@click.option("--account", help="Cloudflare account ID", envvar="CLOUDFLARE_ACCOUNT_ID")
@click.option("--account-name", help="Cloudflare account name", envvar="CLOUDFLARE_ACCOUNT_NAME")
@click.option("--project", help="Cloudflare Pages project name", envvar="CLOUDFLARE_PROJECT_NAME")
@click.argument("path", type=click.Path(exists=True, path_type=Path, readable=True))
def upload(token, account, account_name, project, path):
    cf = CloudflarePages(
        api_token=token,
        account_id=account,
        account_name=account_name,
        project_name=project,
    )
    cf.upload(path)


if __name__ == "__main__":
    cli()
