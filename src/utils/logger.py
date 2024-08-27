#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import logging.config
import os
from typing_extensions import Annotated
from src.__config__ import log_config


# --------------------------------------------------------------
# Configure logging
def getLogger(name: str, level: str = os.getenv("LOG_LEVEL", "DEBUG")):
    logger: Annotated[logging.Logger, "The logger object to use for logging"]
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


if __name__ == "__main__":
    log = getLogger(__name__)
    log.debug("Debug message")

