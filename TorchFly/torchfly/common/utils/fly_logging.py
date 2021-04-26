"""
A logger that maintains logs of both stdout and stderr when models are run.
https://github.com/allenai/allennlp/blob/master/allennlp/common/tee_logger.py
"""
from datetime import datetime
from typing import TextIO
import os
import sys
import logging


def replace_cr_with_newline(message: str):
    """
    TQDM and requests use carriage returns to get the training line to update for each batch
    without adding more lines to the terminal output.  Displaying those in a file won't work
    correctly, so we'll just make sure that each batch shows up on its one line.
    :param message: the message to permute
    :return: the message with carriage returns replaced with newlines
    """
    if "\r" in message:
        message = message.replace("\r", "")
        if not message or message[-1] != "\n":
            message += "\n"
    return message


class TeeLogger:
    """
    This class is an attempt to maintain logs of both stdout and stderr for when models are run.
    To use this class, at the beginning of your script insert these lines::
        sys.stdout = TeeLogger("stdout.log", sys.stdout)
        sys.stderr = TeeLogger("stdout.log", sys.stderr)
    """

    def __init__(
        self, filename: str, terminal: TextIO, file_friendly_terminal_output: bool
    ) -> None:
        self.terminal = terminal
        self.file_friendly_terminal_output = file_friendly_terminal_output
        parent_directory = os.path.dirname(filename)
        os.makedirs(parent_directory, exist_ok=True)
        self.log = open(filename, "a")

    def write(self, message):
        cleaned = replace_cr_with_newline(message)

        if self.file_friendly_terminal_output:
            self.terminal.write(cleaned)
        else:
            self.terminal.write(message)

        self.log.write(cleaned)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        # Mirror the API of sys.stdout so that we can
        # check for the presence of a terminal easily.
        return not self.file_friendly_terminal_output

    def cleanup(self) -> TextIO:
        self.log.close()
        return self.terminal


def init_logging(debug=False):
    # need to reload the old logging
    from imp import reload
    reload(logging)

    if debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(format=format_str, 
                        level=LEVEL)

def setup_logging(foldername: str, filename=None, debug=False):
    # first make the directory
    os.makedirs(foldername, exist_ok=True)
    
    if filename is None:
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H:%M:%S")
        filename = str(timestampStr) + ".log"

    if debug:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO

    filepath = os.path.join(foldername, filename)

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(format=format_str, 
                        level=LEVEL)

    # logger = logging.getLogger()

    # fileHandler = logging.FileHandler(filepath)
    # stdoutHandler = logging.StreamHandler()
    
    # logger.addHandler(fileHandler)
    # logger.addHandler(stdoutHandler)

    # formatter = logging.Formatter(format_str)
    # fileHandler.setFormatter(formatter)
    # stdoutHandler.setFormatter(formatter)

    # sys.stdout = TeeLogger(filepath, sys.stdout, False)
    # sys.stderr = TeeLogger(filepath, sys.stderr, False)

    # stdout_handler = logging.FileHandler(filepath)
