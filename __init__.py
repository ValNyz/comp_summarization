# -*- coding: utf-8 -*-

import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

# Create the Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

if len(logger.handlers) > 0:
    logger.handlers.clear()

# Create the Handler for logging data to a file
logger_handler = RotatingFileHandler('logs\\mylog.log', maxBytes=1024, backupCount=5)
logger_handler.setLevel(logging.DEBUG)

# Create the Handler for logging data to console.
console_handler = StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(name)s - %(threadName)s - %(levelname)s - %(message)s')

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)
console_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
logger.addHandler(logger_handler)
logger.addHandler(console_handler)
