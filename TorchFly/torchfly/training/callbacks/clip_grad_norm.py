import os
import sys
import time
import torch
import logging
from apex import amp
from omegaconf import DictConfig
from typing import Any, Dict

from .events import Events
from .callback import Callback, handle_event

logger = logging.getLogger(__name__)
Trainer = Any


@Callback.register("gradient_clip_norm")
class GradientClipNorm(Callback):
    """
    Clip the gradient based on its norm
    """
    @handle_event(Events.STEP_BEGIN, priority=-100)
    def gradient_clip_norm(self, trainer: Trainer):
        # gradient norm clipping
        if self.config.training.optimization.max_gradient_norm > 0:
            if self.config.training.optimization.fp16:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(trainer.optimizer), self.config.training.optimization.max_gradient_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), self.config.training.optimization.max_gradient_norm
                )
