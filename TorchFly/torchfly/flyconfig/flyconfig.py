from typing import Any, Dict, List
import os
import sys
import copy
import time
import shutil
import logging
import logging.config
from omegaconf import OmegaConf, DictConfig
import argparse
import re

import torchfly.flyconfig
# from .utils import get_config_fullpath, setup_flyconfig

logger = logging.getLogger(__name__)


def init_omegaconf() -> None:
    """ Initilize Omegaconf custom functions"""
    def _time_pattern(pattern: str):
        return time.strftime(pattern, time.localtime())

    try:
        OmegaConf.register_resolver("now", _time_pattern)
    except AssertionError as e:
        logger.warning(e)


def search_config(config_dir, config_name) -> str:
    """ 
    This function searches matching `config_name` in `config_dir`.
    
    Args:
        config_dir: where the config is located (before changing the directory)
        config_name: the name of composed config
    Returns:
        filename: the path of the matched config filename
    """
    yaml_exts = (".yaml", ".yml")
    _, ext = os.path.splitext(config_name)

    # if config_name does not have file extension specified
    if ext == '':
        # search if matching yml or yaml
        for yaml_ext in yaml_exts:
            yaml_file = os.path.join(config_dir, config_name + yaml_ext)
            if os.path.exists(yaml_file):
                return yaml_file
        # if nothing is found
        raise ValueError(f"Cannot find {yaml_file}! Please check if the file exists.")
    # if the file extension is provided
    elif ext in yaml_exts:
        return os.path.join(config_dir, config_name)
    # if not in yaml format
    else:
        raise ValueError(f"Please check if {config_name} is in yaml extension!")


def merge_subconfigs(config_dir: str, subconfigs: List, config: OmegaConf = None) -> OmegaConf:
    """
    This function merges the config's defaults lists and load them into the config

    Args:
        config_dir: where the config is located
        config: existing Omega config
        subconfigs: A list of subconfigs
    Returns:
        new_config: config after merge
    """
    # config is not defined
    if config is None:
        config = OmegaConf.create()

    # search and merge all subconfigs
    for subconfig in subconfigs:
        subconfig_dir = list(subconfig)[0]
        subconfig_name = subconfig[subconfig_dir]
        subconfig_path = search_config(os.path.join(config_dir, subconfig_dir), subconfig_name)
        subconfig = OmegaConf.load(subconfig_path)
        # merge the current config
        config = OmegaConf.merge(config, subconfig)
    return config


def load_config(config_path: str) -> OmegaConf:
    """
    This function loads the user's configuration
    
    Args:
        config_path: The path for the yaml file
    Returns:
        config: OmegaConf conguration
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Cannot find {config_path}!")

    # load user configuration
    config = OmegaConf.load(config_path)
    # get subconfigs
    if "defaults" in config:
        subconfigs = config["defaults"]
        logger.warning("The use of the name `defaults` in config is deprecated! Use `subconfigs` instead!")
    elif "subconfigs" in config:
        subconfigs = config["subconfigs"]
    # if not defined subconfigs
    else:
        subconfigs = []

    # search and merge all sub-configurations
    config = merge_subconfigs(config_dir=os.path.dirname(config_path), config=config, subconfigs=subconfigs)
    return config


def search_argv_config_path():
    """
    This function searches the config path is set in the argument
    """
    for idx, argv in enumerate(sys.argv[1:]):
        if argv.startswith("--config_path") or argv.startswith("--config"):
            try:
                config_path = sys.argv[idx + 2]
            except IndexError:
                raise ValueError("Please provide the path after --config.")

            if not os.path.exists(config_path):
                raise ValueError(f"Cannot find {config_path}! Please specify a valid path.")

            return config_path
    
    # if argv does not provide config_path
    return None


class FlyConfig:
    @staticmethod
    def load(config_path: str = None):
        # register omegaconf resolves
        init_omegaconf()

        # Find config in the command argument
        argv_config_path = search_argv_config_path()
        # Check if config_path is set or not
        if config_path is not None and argv_config_path is not None:
            logger.warning(
                f"Overriding the original config {config_path} with the command-line argument {argv_config_path}"
            )
        elif config_path is None and argv_config_path is None:
            raise ValueError("Please specify `config_path` in the argument or command-line!")
        elif config_path is None and argv_config_path is not None:
            config_path = argv_config_path

        # load user's config
        user_config = load_config(config_path)
        # load system's default config
        flyconfig_module_path = os.path.dirname(os.path.abspath(torchfly.flyconfig.__file__))
        system_config_path = os.path.join(flyconfig_module_path, "config", "flyconfig.yml")
        system_config = load_config(system_config_path)

        # combine user and system config
        config = OmegaConf.merge(system_config, user_config)

        # override configuration from the arguments parsing
        config = override_config(sys.argv[1:], config_dir=os.path.dirname(config_path), config=config)

        # clean subconfigs in config as we don't need it anymore
        if "defaults" in config:
            del config["defaults"]
        elif "subconfigs" in config:
            del config["subconfigs"]

        # set runtime.cwd
        config.flyconfig.runtime.cwd = os.getcwd()
        config.flyconfig.runtime.config_path = config_path

        return config

    @staticmethod
    def print(config):
        new_config = copy.copy(config)
        del new_config["flyconfig"]
        # print(OmegaConf.to_yaml(new_config))
        return OmegaConf.to_yaml(new_config)

    # @staticmethod
    # def to_yaml(config):
    #     new_config = copy.copy(config)
    #     del new_config["flyconfig"]
    #     return OmegaConf.to_yaml(new_config)


def check_valid_arg(arg):
    """
    This function check if arg is a valid override

    Args:
        arg (str): a command-line argument
    Returns:
        valid (bool): True or False
    """
    if re.search(".+=.+", arg) is not None:
        return True
    elif arg.startswith("-") or arg.startswith("--"):
        return False
    else:
        # logger.warning(f"{arg} cannot be parsed!")
        return False


def override_config(argv: List[str], config_dir, config) -> OmegaConf:
    """
    This function overrides the config and the subconfigs from the argvs
    """
    valid_args = []
    for arg in argv:
        if check_valid_arg(arg):
            valid_args.append(arg)

    # get all subconfig overrides
    config_subconfig_dirs = []
    if "subconfigs" in config:
        config_subconfig_dirs = [list(subconfig)[0] for subconfig in config["subconfigs"]]
    elif "defaults" in config:
        config_subconfig_dirs = [list(subconfig)[0] for subconfig in config["defaults"]]

    other_args = []
    valid_subconfigs = []
    for arg in valid_args:
        # check if the argument is overriding the subconfig
        subconfig_dir = arg.split("=")[0]
        if "." not in subconfig_dir:
            # check if subconfig_dir is defined in config_subconfig_dirs
            if subconfig_dir in config_subconfig_dirs:
                subconfig_name = arg.split("=")[1]
                valid_subconfigs.append({subconfig_dir: subconfig_name})
                continue

        other_args.append(arg)

    # get the root subconfigs overrided
    config = merge_subconfigs(config_dir, valid_subconfigs, config)

    # override the rest config
    arg_config = OmegaConf.from_dotlist(other_args)
    config = OmegaConf.merge(config, arg_config)

    return config