#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
import logging
import logging.config
import os
from typing_extensions import Annotated
from src.__config__ import log_config


# --------------------------------------------------------------
# Configure logging
def getLogger(name: str, level: str = os.getenv("LOG_LEVEL", "DEBUG")):
    logger: Annotated[logging.Logger, 'The logger object']
    logging.config.dictConfig(log_config)
    caller_filename = get_calling_filename()
    if name == '__main__':
        logger = logging.getLogger(name=name)
        # logger.debug(f"Loggers: {logger.name}")
    elif caller_filename.__contains__(name):
        logger = logging.getLogger(name=name)
        # logger.debug(f"Loggers: {logger.name}")
    else:
        logger = logging.getLogger(caller_filename.split('/')[-1])
        # logger.debug(f"Loggers: {logger.name}")
    logger.setLevel(level)
    reduce_log_dir(5, logger)

    return logger



def get_calling_filename():
    stack = traceback.extract_stack()
    # The second-to-last frame is the caller
    calling_frame = stack[-2]
    return calling_frame.filename


def reduce_log_dir(size: int, logger: logging.Logger):
    log_dir = os.getenv("LOG_DIR")
    if log_dir:
        logs = os.listdir(log_dir)
        if len(logs) > size:
            logs.sort()
            for i in range(len(logs) - size):
                log_file = os.path.join(log_dir, logs[i])
                os.remove(log_file)
                # logger.debug(f"Removed log file: {log_file}")
    else:
        logger.debug("No log directory specified")
# # Example usage
# if __name__ == "__main__":
#     log = getLogger(__name__.__class__.__str__('utf-8'))
#     log.debug("Debug message")
#     log.debug(f"Calling filename: {get_calling_filename()}")

