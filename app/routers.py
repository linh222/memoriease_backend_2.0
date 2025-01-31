import logging
from importlib import import_module

logger = logging.getLogger(__name__)

app_routers = [
    "app.apis.general.apis",
    "app.apis.prediction.apis",
    "app.apis.image_by_day.apis",
    "app.apis.metadata.apis",
]


def load_routers(app):
    for path in app_routers:
        module = import_module(path)

        function = getattr(module, "include_router")
        function(app)

        logger.info("Initialized router to %s", path)
