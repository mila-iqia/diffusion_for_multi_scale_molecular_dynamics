import logging
import sys
import time
from logging.handlers import WatchedFileHandler
from pathlib import Path

LOG_FILE_NAME = "active_learning.log"


class _CampaignLogContext:
    """Mutable holder for the current epoch and stage, injected into each log line's prefix."""

    def __init__(self):
        self.epoch = None
        self.stage = None

    def set(self, epoch: int, stage: str) -> None:
        """Record the epoch/stage now running (shown on every subsequent log line)."""
        self.epoch = epoch
        self.stage = stage

    def clear(self) -> None:
        """Forget the epoch/stage (log lines then carry only the timestamp)."""
        self.epoch = None
        self.stage = None


# The active learning loop updates this as it moves through epochs and stages; the formatter reads it.
CAMPAIGN_LOG_CONTEXT = _CampaignLogContext()


class _CampaignFormatter(logging.Formatter):
    """Prefix each line with the timestamp and, once a stage is running, the epoch and stage."""

    def __init__(self):
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Render '<timestamp> :: <message>', or '<timestamp> | Epoch N Stage :: <message>'."""
        timestamp = self.formatTime(record, self.datefmt)
        message = record.getMessage()
        if CAMPAIGN_LOG_CONTEXT.epoch is None:
            return f"{timestamp} :: {message}"
        return f"{timestamp} | Epoch {CAMPAIGN_LOG_CONTEXT.epoch} {CAMPAIGN_LOG_CONTEXT.stage} :: {message}"


class _DropDependencyChatterFilter(logging.Filter):
    """Drop sub-warning log records from noisy dependencies (e.g. maml's 'Structure index N is rotated').

    maml sets its own logger to INFO at import (after our setup), so a parent setLevel does not stick; a
    handler filter reliably keeps the noise out of the campaign log while still letting warnings through.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep the record unless it is a sub-warning message from maml."""
        return not (record.name.startswith("maml") and record.levelno < logging.WARNING)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def set_up_campaign_logger(working_directory: Path):
    """Set up the campaign logger writing to '<working_directory>/active_learning.log' and to stdout.

    A blank line is prepended when the log file already exists so a restart is visually separated from
    the previous run.

    Args:
        working_directory: Path to the working directory where the log will be written.

    Returns:
        logger: a configured logger.
    """
    log_file = working_directory / LOG_FILE_NAME
    if log_file.is_file() and log_file.stat().st_size > 0:
        with open(log_file, "a") as file_descriptor:
            file_descriptor.write("\n")  # separate this (re)start from the previous run

    CAMPAIGN_LOG_CONTEXT.clear()

    logger = logging.getLogger()
    logging.captureWarnings(capture=True)
    logger.setLevel(logging.INFO)

    formatter = _CampaignFormatter()
    dependency_filter = _DropDependencyChatterFilter()
    file_handler = WatchedFileHandler(log_file)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        handler.addFilter(dependency_filter)
        logger.addHandler(handler)

    logger.propagate = False  # Prevent messages from propagating to root logger

    return logger


def clean_up_campaign_logger(logger: logging.Logger):
    """Remove the logger."""
    CAMPAIGN_LOG_CONTEXT.clear()
    for handler in list(logger.handlers):
        handler.close()  # Close the file handler to release the file
        logger.removeHandler(handler)
