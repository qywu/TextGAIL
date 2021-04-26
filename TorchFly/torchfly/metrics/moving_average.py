import logging
import math
import numpy as np
from overrides import overrides

from .metric import Metric

# logger = logging.getLogger(__name__)


@Metric.register("moving_average")
class MovingAverage(Metric):
    """
    This :class:`Metric` breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """
    def __init__(self, beta: float = 0.9, name="moving_average") -> None:
        super().__init__(name)
        self.beta = beta
        self._avg_value = -99999
        self.logger = logging.getLogger(name) 

    @overrides
    def __call__(self, value):
        """
        Args:
            value : `float` The value to average.
        """
        if math.isnan(value):
            self.logger.warn(f"Detected nan in {self.name}! Skipping the value")
        elif isinstance(value, float):
            if self._avg_value == -99999:
                self._avg_value = value
            else:
                self._avg_value = self.beta * self._avg_value + (1 - self.beta) * value
        else:
            self.logger.error(f"Cannot tale {type(value)} type")
            raise NotImplementedError

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns:
            The moving average value.
        """
        value = self._avg_value
        if reset:
            self.reset()
        return value

    @overrides
    def reset(self):
        self._avg_value = -99999