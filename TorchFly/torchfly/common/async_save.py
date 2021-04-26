import torch
from torch.multiprocessing import Process
import numpy as np
import logging
from collections import OrderedDict
from typing import Any, Dict

logger = logging.getLogger(__name__)


def copy_cpu_state_dict(states: Any) -> Dict[str, Any]:
    # need a new dict
    result_states = OrderedDict()

    if isinstance(states, dict):
        # recursion
        for k in states:
            result_states[k] = copy_cpu_state_dict(states[k])
    elif isinstance(states, list):
        result_states = [copy_cpu_state_dict(item) for item in states]
    elif isinstance(states, torch.Tensor):
        # If it is torch.Tensor, copy to cpu first
        result_states = states.cpu()
    elif isinstance(states, (int, float, str, tuple, type(None))):
        result_states = states
    else:
        result_states = states
        logging.warn(f"`copy_cpu_state_dict` cannot parse {type(states)}")
        # print(f"`copy_cpu_state_dict` cannot parse {type(states)}")
    return result_states


def _save(states: OrderedDict, filename):
    torch.save(states, filename)
    return 0


def async_save(model_states: OrderedDict, filename) -> Process:
    model_states = copy_cpu_state_dict(model_states)
    p = Process(target=_save, args=(model_states, filename), daemon=True)
    p.daemon = True
    p.start()
    return p


def async_wait(process: Process):
    process.join()