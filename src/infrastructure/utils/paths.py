from src.share.settings.app import Environment, app_settings


def get_model_root_path() -> str:
    model_root = ''
    match app_settings.ENVIRONMENT:
        case Environment.CONTAINER:
            model_root = "/embedding_deployment"
        case Environment.LOCAL_TEST:
            model_root = "./models"
        case _:
            raise Exception(f"Invalid environment option: {app_settings.ENVIRONMENT}")

    return model_root