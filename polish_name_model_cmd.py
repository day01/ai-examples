import click
from config.polish_name import PolishNameConfig
from polish_name_data_loader import PolishNameDataLoader
from polish_name_model import PolishNameModel


@click.command(name="polish-name-model")
@click.option("--ignore", is_flag=True, help="Ignore refreshing data")
def polish_name_model_cmd(ignore):
    """This is the command for running the Polish Name Model"""
    config = PolishNameConfig.load_config("./config/polish_name_config.yaml")
    data_loader = PolishNameDataLoader(config)

    if not ignore:
        data_loader.refresh_data()

    male, female = data_loader.load_data()
    model = PolishNameModel(config)
    model.build_model(male)
    model.train_model()
    model.train_loss()
    model.dev_loss()
    model.test_loss()

    model.display_dimension()

    for index in range(10):
        model.predict(index)
    click.echo("Polish Name Model has been run successfully")


if __name__ == "__main__":
    polish_name_model_cmd()
