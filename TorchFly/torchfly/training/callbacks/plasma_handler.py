import os
import sys
import time
import torch
import logging
from apex import amp
from omegaconf import DictConfig
from typing import Any, Dict
from torchfly.training.data.plasma import init_plasma, get_plasma_manager

from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger(__name__)
Trainer = Any

@Callback.register("plasma_handler")
class PlasmaHandler(Callback):

    @handle_event(Events.INITIALIZE, priority=200)
    def start_plasma(self, trainer: Trainer):
        init_plasma()
        self.plasma_store_address = get_plasma_manager().plasma_store_address
        logger.info("Plasma Started!")

    @handle_event(Events.TRAIN_BEGIN, priority=200)
    def set_plasma_store_address(self, trainer: Trainer):
        trainer.config.plasma.plasma_store_address = self.plasma_store_address