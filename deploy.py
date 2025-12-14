from src.infrastructure.image_builders import get_image_builder
from src.settings import deployment_settings


image_builder = get_image_builder(deployment_settings.DEPLOYMENT_OPTION)

image_builder.build_image(deployment_settings.QUANTIZATION)