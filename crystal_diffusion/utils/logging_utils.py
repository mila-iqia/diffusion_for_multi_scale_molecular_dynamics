import logging
import os
import socket
import sys
from logging import StreamHandler
from logging.handlers import WatchedFileHandler

from git import InvalidGitRepositoryError, Repo
from pip._internal.operations import freeze

logger = logging.getLogger(__name__)


def setup_console_logger(experiment_dir: str):
    """Set up the console logger.

    This method sets up logging for analysis scripts. It is very opinionated about how to log:
        - Always log to a file, with a fixed name, in the experiment_dir folder.
        - Always log to console as well.

    Args:
        experiment_dir : directory where the logging file will be written.

    Returns:
        no output.
    """
    logging.captureWarnings(capture=True)

    logging_format = "%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove stale handlers. Stale handlers can occur when this setup method is called multiple times,
    # in tests for examples.
    for handler in root.handlers:
        if type(handler) is WatchedFileHandler or type(handler) is StreamHandler:
            root.removeHandler(handler)

    console_log_file = os.path.join(experiment_dir, "console.log")
    formatter = logging.Formatter(logging_format)

    file_handler = WatchedFileHandler(console_log_file)
    stream_handler = StreamHandler(stream=sys.stdout)

    list_handlers = [file_handler, stream_handler]

    for handler in list_handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)


def setup_analysis_logger():
    """This method sets up logging for analysis scripts."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    analysis_logging_format = "%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - %(message)s"
    formatter = logging.Formatter(analysis_logging_format)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


class LoggerWriter:
    """LoggerWriter.

    see: https://stackoverflow.com/questions/19425736/
    how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, printer):
        """__init__.

        Args:
            printer: (fn) function used to print message (e.g., logger.info).
        """
        self.printer = printer
        self.encoding = None

    def write(self, message):
        """write.

        Args:
            message: (str) message to print.
        """
        if message != '\n':
            self.printer(message)

    def flush(self):
        """flush."""
        pass


def get_git_hash(script_location):
    """Find the git hash for the running repository.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :return: (str) the git hash for the repository of the provided script.
    """
    if not script_location.endswith('.py'):
        raise ValueError('script_location should point to a python script')
    repo_folder = os.path.dirname(script_location)
    try:
        repo = Repo(repo_folder, search_parent_directories=True)
        commit_hash = repo.head.commit
    except (InvalidGitRepositoryError, ValueError):
        commit_hash = 'git repository not found'
    return commit_hash


def log_exp_details(script_location, args):
    """Will log the experiment details to both screen logger and mlflow.

    :param script_location: (str) path to the script inside the git repos we want to find.
    :param args: the argparser object.
    """
    git_hash = get_git_hash(script_location)
    hostname = socket.gethostname()
    dependencies = freeze.freeze()
    details = "\nhostname: {}\ngit code hash: {}\ndata folder: {}\ndata folder (abs): {}\n\n" \
              "dependencies:\n{}".format(
                  hostname, git_hash, args.data, os.path.abspath(args.data),
                  '\n'.join(dependencies))
    logger.info('Experiment info:' + details + '\n')
