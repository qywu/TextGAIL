import os
import sys
import copy
import time
import shutil
import logging
import logging.config
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, List
import argparse

from .utils import get_config_fullpath, setup_flyconfig

logger = logging.getLogger(__name__)


class Singleton(type):
    """A metaclass that creates a Singleton base class when called."""
    _instances: Dict[type, "Singleton"] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def init_omegaconf() -> None:
    """Initilize Omegaconf custom functions"""
    def _time_pattern(pattern: str):
        return time.strftime(pattern, time.localtime())

    try:
        OmegaConf.register_resolver("now", _time_pattern)
    except AssertionError as e:
        logger.warning(e)


class GlobalFlyConfig(metaclass=Singleton):
    def __init__(self, config_path: str = None, disable_chdir: bool = False, disable_logging: bool = False):
        """Initialize FlyConfig globally

        Args:
            config_path: where the config file is located.
            disable_chdir: since FlyConfig changes working dir, this argument can disable it
        """
        self.disable_chdir = disable_chdir
        self.disable_logging = disable_logging
        self.initialized = False
        self.user_config = None
        self.system_config = None
        self.old_cwd = os.getcwd()

        config_path = get_config_path_from_argv(config_path)

        if config_path is None:
            raise ValueError(
                "Please provide config_path via argument `--config_path your_file` or when initializing GlobalFlyConfig!"
            )

        self.initialize(config_path)

    def initialize(self, config_path: str, force: bool = False) -> OmegaConf:
        """
        Args:
            config_path: a file or dir
            force: ignore if initialized
        Returns:
            user_config: only return the user config
        """

        if self.initialized and not force:
            raise ValueError("FlyConfig is already initialized!")

        config_path = check_config_path(config_path)

        init_omegaconf()

        system_config = load_system_config()
        user_config = load_user_config(config_path)

        config = OmegaConf.merge(system_config, user_config)

        # get current working dir
        config.flyconfig.runtime.cwd = os.getcwd()

        # change working dir
        if not self.disable_chdir:
            working_dir_path = config.flyconfig.run.dir
            os.makedirs(working_dir_path, exist_ok=True)
            os.chdir(working_dir_path)

        # configure logging
        if int(os.environ.get("LOCAL_RANK", 0)) == 0 and not self.disable_logging:
            logging.config.dictConfig(OmegaConf.to_container(config.flyconfig.logging))
            logger.info("FlyConfig Initialized")
            if not self.disable_chdir:
                logger.info(f"Working directory is changed to {working_dir_path}")

        # clean defaults
        if "defaults" in config:
            del config["defaults"]
        elif "subconfigs" in config:
            del config["subconfigs"]

        # overrides
        overrides = get_overrides_from_argv(sys.argv[1:])
        overrides_config = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, overrides_config)

        # get system config
        self.system_config = OmegaConf.create({"flyconfig": OmegaConf.to_container(config.flyconfig)})

        # get user config
        self.user_config = copy.deepcopy(config)
        del self.user_config["flyconfig"]

        # save config
        if int(os.environ.get("LOCAL_RANK", 0)) == 0 and not self.disable_logging:
            os.makedirs(self.system_config.flyconfig.output_subdir, exist_ok=True)

            # save the entire config directory
            cwd = self.system_config.flyconfig.runtime.cwd
            dirpath = os.path.join(cwd, os.path.dirname(config_path))
            dst_config_path = os.path.join(self.system_config.flyconfig.output_subdir, "config")

            if not os.path.exists(dst_config_path):
                shutil.copytree(dirpath, dst_config_path)

                # save system config
                _save_config(
                    filepath=os.path.join(self.system_config.flyconfig.output_subdir, "flyconfig.yml"),
                    config=self.system_config
                )

                # save user config
                _save_config(
                    filepath=os.path.join(self.system_config.flyconfig.output_subdir, "config.yml"),
                    config=self.user_config
                )

                logger.info("\n\nConfiguration:\n" + OmegaConf.to_yaml(self.user_config))

        self.initialized = True

        return self.user_config

    def is_initialized(self) -> bool:
        return self.initialized

    def clear(self) -> None:
        self.initialized = False
        self.config = None


def get_config_path_from_argv(config_path):
    argv_config_path = None
    for idx, argv in enumerate(sys.argv[1:]):
        if argv.startswith("--config_path") or argv.startswith("--config"):
            try:
                argv_config_path = sys.argv[idx + 2]
            except IndexError:
                raise ValueError("Please provide the path after --config.")
            argv_config_path = check_config_path(argv_config_path)
            break

    if argv_config_path is None:
        return config_path
    else:
        if config_path is not None:
            logger.warning("Overriding the old config_path from --config!")
        return argv_config_path


def check_config_path(config_path) -> str:
    # Search config file
    if os.path.isdir(config_path):
        if os.path.exists(os.path.join(config_path, "config.yaml")):
            config_file = "config.yaml"
        elif os.path.exists(os.path.join(config_path, "config.yml")):
            config_file = "config.yml"
        else:
            raise ValueError("Cannot find config.yml. Please specify `config_file`")
        config_path = os.path.join(config_path, config_file)
    else:
        if not os.path.isfile(config_path):
            raise ValueError("Please provide a valid config path.")
    return config_path


def get_overrides_from_argv(arguments: List[str]) -> List:
    overrides = []
    for argv in arguments:
        if "=" in argv and not argv.startswith("-") and not argv.startswith("--"):
            overrides.append(argv)
            # arguments.remove(argv)
    return overrides


def _save_config(filepath, config):
    "Save the config file"
    with open(filepath, "w") as f:
        OmegaConf.save(config, f)


def merge_defaults(config_dir: str, config: OmegaConf, defaults: List) -> OmegaConf:
    """
    Merge the default lists and put into the config

    Args:
        config_dir: where the config is located
        config: existing Omega config
        defaults: A list of default items
    Returns:
        new_config: config after merge
    """
    for default in defaults:
        subconfig_key = list(default)[0]
        subconfig_value = default[subconfig_key]

        subconfig_fullpath = get_config_fullpath(os.path.join(config_dir, subconfig_key), subconfig_value)
        subconfig = OmegaConf.load(subconfig_fullpath)

        config = OmegaConf.merge(config, subconfig)

    return config


def load_system_config() -> OmegaConf:
    module_path = os.path.dirname(os.path.abspath(__file__))
    system_config_path = os.path.join(module_path, "config", "flyconfig.yml")
    system_config = OmegaConf.load(system_config_path)
    system_defaults = system_config["subconfigs"]

    sysmte_config = merge_defaults(
        config_dir=os.path.dirname(system_config_path), config=system_config, defaults=system_defaults
    )
    return sysmte_config


def load_user_config(config_path: str) -> OmegaConf:
    if not os.path.exists(config_path):
        raise ValueError(f"Cannot find {config_path}")

    user_config = OmegaConf.load(config_path)
    user_defaults = user_config["defaults"]

    user_config = merge_defaults(config_dir=os.path.dirname(config_path), config=user_config, defaults=user_defaults)
    return user_config
