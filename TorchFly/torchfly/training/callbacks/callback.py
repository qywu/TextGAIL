import logging
from omegaconf import DictConfig
from typing import Any, Callable

from .events import Events
from ...common.registrable import Registrable

logger = logging.getLogger(__name__)
Trainer = Any

__all__ = ["Callback", "handle_event"]

def handle_event(event: str, priority: int = 0):
    def wrapper(method: Callable[[], None]):
        setattr(method, "_event", event)
        setattr(method, "_priority", priority)
        return method

    return wrapper


class Callback(Registrable):
    """
    Base class for callbacks that want to record values, dynamically change learner params, etc.
    It is the programer's responsibility to handler distributed setup here.
    
    Attributes:
        trainer.__master: bool (default = True) It is used in distributed training to identify the master
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    @handle_event(Events.INITIALIZE)
    def on_initialize(self, trainer: Trainer) -> None:
        "To initialize before the distributed spawning."
        pass

    @handle_event(Events.TRAIN_BEGIN)
    def on_train_begin(self, trainer: Trainer) -> None:
        "To initialize constants in the callback. This is after the distributed spawning."
        pass

    @handle_event(Events.EPOCH_BEGIN)
    def on_epoch_begin(self, trainer: Trainer) -> None:
        "At the beginning of each epoch."
        pass

    @handle_event(Events.BATCH_BEGIN)
    def on_batch_begin(self, trainer: Trainer) -> None:
        "Set HP before the output and loss are computed."
        pass

    @handle_event(Events.LOSS_BEGIN)
    def on_loss_begin(self, trainer: Trainer) -> None:
        "Called after forward pass but before loss has been computed."
        pass

    @handle_event(Events.BACKWARD_BEGIN)
    def on_backward_begin(self, trainer: Trainer) -> None:
        "Called after the forward pass and the loss has been computed, but before backprop."
        pass

    @handle_event(Events.BACKWARD_END)
    def on_backward_end(self, trainer: Trainer) -> None:
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass

    @handle_event(Events.STEP_BEGIN)
    def on_step_begin(self, trainer: Trainer) -> None:
        "Called before the step of the optimizer."
        pass

    @handle_event(Events.STEP_END)
    def on_step_end(self, trainer: Trainer) -> None:
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass

    @handle_event(Events.BATCH_END)
    def on_batch_end(self, trainer: Trainer) -> None:
        "Called at the end of the batch."
        pass

    @handle_event(Events.EPOCH_END)
    def on_epoch_end(self, trainer: Trainer) -> None:
        "Called at the end of an epoch."
        pass

    @handle_event(Events.VALIDATE_BEGIN)
    def on_validate_begin(self, trainer: Trainer) -> None:
        "Called at the beginning of validation"
        pass

    @handle_event(Events.VALIDATE_END)
    def on_validate_end(self, trainer: Trainer) -> None:
        "Called at the end of validation"
        pass

    @handle_event(Events.TRAIN_END)
    def on_train_end(self, trainer: Trainer) -> None:
        "Useful for cleaning up things and saving files/models."
        pass

    def state_dict(self) -> dict:
        """
        If this callback contains state that should be checkpointed for training,
        return it here (with a key that's unique to this callback).
        If the state lives in a pytorch object with a `state_dict`
        method, this should return the output of `state_dict()`, not the object itself.
        This default implementation suffices when there's no state to checkpoint.
        """
        return {}

    def load_state_dict(self, training_state: dict) -> None:
        """
        Given a dict of training state, pull out the relevant parts
        and rehydrate the state of this callback however is necessary.
        This default implementation suffices when there's no state to restore.
        """
        pass