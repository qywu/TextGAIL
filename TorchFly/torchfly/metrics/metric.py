import torch
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..common.registrable import Registrable


class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated. (allennlp/training/metrics/metric.py)
    """
    def __init__(self, name=None):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        The metric value must be numeric.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)