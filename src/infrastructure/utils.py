from src.infrastructure.enums import Environment
from src.share.settings.environment import EnvironmentSettings


def get_model_root_path():
    environment_settings = EnvironmentSettings()

    match environment_settings.ENVIRONMENT:
        case Environment.CONTAINER:
            return "/embedding_deployment"
        case Environment.LOCAL:
            return "./models"
        case _:
            raise Exception("Invalid environment!")
