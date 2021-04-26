import inspect
import logging
from collections import defaultdict
from omegaconf import DictConfig
from typing import Callable, Dict, List, NamedTuple

from .callback import Callback

logger = logging.getLogger(__name__)

__all__ = ["CallbackHandler"]

def _is_event_handler(member) -> bool:
    return inspect.ismethod(member) and hasattr(member, "_event") and hasattr(member, "_priority")


class EventHandler(NamedTuple):
    name: str
    callback: Callback
    handler: Callable
    priority: int


class CallbackHandler:
    def __init__(self, config: DictConfig, trainer, callbacks=None, verbose=False):
        """
        Args:
            verbose : bool, optional (default = False) Used for debugging.
        """
        self.trainer = trainer
        self.events: Dict[str, List[EventHandler]] = defaultdict(list)
        self.callbacks = []

        for callback in callbacks:
            self.add_callback(callback)

        self.verbose = verbose

    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)

        for name, method in inspect.getmembers(callback, _is_event_handler):
            event = getattr(method, "_event")
            priority = getattr(method, "_priority")
            self.events[event].append(EventHandler(name, callback, method, priority))
            self.events[event].sort(key=lambda _evt: _evt.priority, reverse=True)

    def fire_event(self, event: str) -> None:
        """
        Runs every callback registered for the provided event,
        ordered by their priorities.
        """
        for event_handler in self.events.get(event, []):
            if self.verbose:
                logger.debug(f"event {event} -> {event_handler.name}")
            event_handler.handler(self.trainer)

    def list_events(self) -> List[Callback]:
        """
        Returns the callbacks associated with this handler.
        Each callback may be registered under multiple events,
        but we make sure to only return it once. If `typ` is specified,
        only returns callbacks of that type.
        """
        return list({event.callback for event_list in self.events.values() for event in event_list})
