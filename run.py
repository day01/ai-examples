import click

from polish_name_model_cmd import polish_name_model_cmd


@click.group()
def cli():
    pass


cli.add_command(polish_name_model_cmd)


if __name__ == "__main__":
    cli()
