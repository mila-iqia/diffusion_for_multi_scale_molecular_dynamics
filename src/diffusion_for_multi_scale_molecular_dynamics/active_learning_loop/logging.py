import logging
import random
import string
import sys
from logging.handlers import WatchedFileHandler
from pathlib import Path


def generate_random_string(size: int):
    """Generate a random string."""
    chars = string.ascii_uppercase + string.ascii_lowercase
    return "".join(random.choice(chars) for _ in range(size))


def set_up_campaign_logger(working_directory: Path):
    """Set up campaign logger.

    Sets up a unique logger for a given iteration. Ensures that this logger logs only to its specific file.

    Args:
        working_directory: Path to the working directory where the log will be written.

    Returns:
        logger: a configured logger.
    """
    log_file_name = "active_learning.log"
    logger_name = generate_random_string(size=16)  # Get a unique name to avoid overlogging.
    logger = logging.getLogger(logger_name)

    logging.captureWarnings(capture=True)
    logging_format = ("%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - %(message)s")
    logger.setLevel(logging.INFO)

    console_log_file = working_directory / log_file_name
    formatter = logging.Formatter(logging_format)

    file_handler = WatchedFileHandler(console_log_file)
    stream_handler = logging.StreamHandler(stream=sys.stdout)

    list_handlers = [file_handler, stream_handler]

    for handler in list_handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False  # Prevent messages from propagating to root logger

    return logger


def clean_up_campaign_logger(logger: logging.Logger):
    """Remove the logger."""
    for handler in list(logger.handlers):  # Iterate over a copy to safely remove
        handler.close()  # Close the file handler to release the file
        logger.removeHandler(handler)
