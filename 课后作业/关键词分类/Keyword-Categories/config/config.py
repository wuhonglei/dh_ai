import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


class BaseConfig:
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
            handlers=[
                logging.StreamHandler(),
                RotatingFileHandler(
                    datetime.now().strftime("./log/%Y-%m-%d.log"),
                    maxBytes=1000000,
                    backupCount=1,
                ),
            ],
        )


settings = BaseConfig()
settings.setup_logging()
