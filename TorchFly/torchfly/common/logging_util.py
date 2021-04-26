import os
import sys
import logging
import colorlog
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def configure_logging(config: DictConfig = None) -> None:
    """
    This function initializes the logging. It is recommended to use Hydra to 
    configure the training and pass the config to this function.

    Args:
        config: A DictConfig from hydra.main
    """
    if config is None:
        config = DictConfig(
            {
                "logging": {
                    "log_dir": "logs",
                    "level": "INFO",
                    "color": True,
                },
                "training": {
                    "rank": 0,
                    "num_gpus_per_node": 1,
                },
            }
        )
    elif config.logging.log_dir is None:
        log_dir = "logs"
    else:
        log_dir = config.logging.log_dir

    os.makedirs(log_dir, exist_ok=True)

    # Only setup training for node 0
    if not hasattr(config.training, "rank") or config.training.rank == 0 or config.training.rank is None:
        root = logging.getLogger()
        root.setLevel(getattr(logging, config.logging.level))
        # setup formaters
        file_formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        if config.logging.color:
            stream_formater = colorlog.ColoredFormatter(
                "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
            )
        else:
            stream_formater = file_formatter
        # setup handlers
        if config.training.num_gpus_per_node > 1:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(stream_formater)
            root.addHandler(stream_handler)

        # append the log
        file_handler = logging.FileHandler(os.path.join(log_dir, f"experiment.log"), mode='a')
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)


# def get_original_cwd(config, resume_mode) -> str:
#     if resume_mode:
#         os.getcwd()
#     else:
#         return os.getcwd()