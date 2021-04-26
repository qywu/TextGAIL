from typing import Any, Dict
import os
import sys
import torch
import termios
import logging
import atexit
from omegaconf import DictConfig

from ..events import Events
from ..callback import Callback, handle_event
from ....utils.distributed import get_rank

logger = logging.getLogger(__name__)

Trainer = Any


@Callback.register("console")
class Console(Callback):
    """
    Callback that interrupt the training and open the console for user interactions
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        if get_rank() != 0:
            raise NotImplementedError("Console can only be called by the main rank 0!")

        # initialize the terminal
        self.old_settings = termios.tcgetattr(sys.stdin)
        self.listener_mode = True
        atexit.register(self.on_exit)

    @handle_event(Events.BATCH_BEGIN)
    def check_interrupt(self, trainer: Trainer):
        key = os.read(sys.stdin.fileno(), 1)

        if key != b'' and key != None:

            if key == b'~':
                self.listener_mode = False
                self.call_console(trainer)
                self.listener_mode = True

            # clear stdin buffer
            while key != b'' and key != None:
                key = os.read(sys.stdin.fileno(), 1)

    @property
    def listener_mode(self):
        return self._listener_mode

    @listener_mode.setter
    def listener_mode(self, value):
        if not isinstance(value, bool):
            raise NotImplementedError("Not boolean type!")

        if not hasattr(self, "_listener_mode"):
            self._listener_mode = False

        if value == True and self._listener_mode == False:
            self.old_settings = termios.tcgetattr(sys.stdin)
            new_settings = termios.tcgetattr(sys.stdin)
            new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)  # lflags
            new_settings[6][termios.VMIN] = 0  # cc
            new_settings[6][termios.VTIME] = 0  # cc
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
        elif value == False and self._listener_mode == True:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

        self._listener_mode = value

    def call_console(self, trainer: Trainer):
        # TODO: Build a more interactive system
        if torch.distributed.is_initialized():
            logger.warning("Detect distributed training! Please be careful about the nccl timeout! (Default is 30 minutes)")
        breakpoint()

    def on_exit(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)