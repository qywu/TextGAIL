import logging
import numpy as np
from overrides import overrides

from .metric import Metric

logger = logging.getLogger(__name__)


@Metric.register("average")
class Average(Metric):
    """
    This :class:`Metric` breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """
    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value):
        """
        Args:
            value : `float` The value to average.
        """
        if isinstance(value, list):
            self._total_value += np.sum(value)
            self._count += len(value)
        elif isinstance(value, float):
            self._total_value += value
            self._count += 1
        else:
            logger.error(f"Cannot tale {type(value)} type")
            raise NotImplementedError

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns:
            The average of all values that were passed to `__call__`.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0