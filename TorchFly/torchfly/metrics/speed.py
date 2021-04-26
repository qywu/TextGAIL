import time
import logging
import numpy as np
from overrides import overrides

from torchfly.metrics.metric import Metric

logger = logging.getLogger(__name__)


@Metric.register("speed")
class Speed(Metric):
    """
    This :class:`Metric` breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """
    def __init__(self, name: str = None) -> None:
        self.name = name
        self._count = 0.0
        self._last_time = time.time()

    @overrides
    def __call__(self, value):
        """
        Args:
            value (float)
        """
        self._count += value

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns:
            The average of all values that were passed to `__call__`.
        """
        now_time = time.time()
        elapsed_time = now_time - self._last_time
        self._last_time = now_time
        speed = self._count / elapsed_time
        self.reset()
        return speed

    @overrides
    def reset(self):
        self._count = 0.0