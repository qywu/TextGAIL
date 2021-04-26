import os
import time
from omegaconf import OmegaConf

import logging

logger = logging.getLogger(__name__)


def setup_flyconfig() -> None:
    """Initilize Omegaconf custom functions"""
    def _time_pattern(pattern: str):
        return time.strftime(pattern, time.localtime())

    try:
        OmegaConf.register_resolver("now", _time_pattern)
    except AssertionError as e:
        logger.warning(e)


def get_config_fullpath(config_dir, config_name) -> str:
    """
    Args:
        config_dir: where the config is located (before changing the directory)
        config_name: the name of composed config
    Returns:
        filename: fullpath of the filename
    """
    yaml_exts = [".yml", ".yaml"]

    filename, ext = os.path.splitext(config_name)

    if ext not in yaml_exts:
        for yaml_ext in yaml_exts:
            yaml_file = os.path.join(config_dir, config_name + yaml_ext)

            if os.path.exists(yaml_file):
                return yaml_file

        # if nothing is found
        raise ValueError(f"Cannot get the full path of {yaml_file}!")
    else:
        return os.path.join(config_dir, config_name)


def split_config_path(config_path):
    if config_path is None or config_path == "":
        return None, None
    root, ext = os.path.splitext(config_path)

    if ext in (".yaml", ".yml"):
        # assuming dir/config.yaml form
        config_file = os.path.basename(config_path)
        config_dir = os.path.dirname(config_path)
    else:
        # assuming dir form without a config file.
        config_file = None
        config_dir = config_path

    if config_dir == "":
        config_dir = None

    if config_file == "":
        config_file = None
    return config_dir, config_file


# def init_flyconfig(config_path):
#     config_dir, config_file = split_config_path(config_path)