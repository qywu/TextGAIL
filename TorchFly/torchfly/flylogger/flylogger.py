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
import torch

import torchfly.utils.distributed as distributed

logger = logging.getLogger(__name__)


class Singleton(type):
    """A metaclass that creates a Singleton base class when called."""
    _instances: Dict[type, "Singleton"] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FlyLogger(metaclass=Singleton):
    """
    FlyLogger sets up the logger and output directory. It is a Singleton class that should be initialized once.

    """
    def __init__(self, config: OmegaConf, logging: bool = True, chdir: bool = True):
        """
        Initialize FlyLogger

        Args:
            config (OmegaConf): loaded by FlyConfig
            logging (bool): whether to setup the logger
            chdir (bool): whether to setup a new directory as in config
        """
        self.config = config
        self.logging = logging
        self.chdir = chdir
        self.initialized = False

        if torch.distributed.is_initialized():
            self.rank = distributed.get_rank()
        else:
            self.rank = int(os.environ.get("RANK", 0))
        
        self.initialize()

    def initialize(self):
        if self.initialized:
            raise ValueError("FlyLogger is already initialized! Please use `.clear()` before initialize it again.")

        self.config.flyconfig.runtime.cwd = os.getcwd()

        # change the directory as in config
        if self.chdir:
            working_dir_path = self.config.flyconfig.run.dir
            os.makedirs(working_dir_path, exist_ok=True)
            os.chdir(working_dir_path)

        # configure logging, FlyLogger should only configure rank 0
        # other ranks should use their own logger
        if self.rank == 0 and self.logging:
            logging.config.dictConfig(OmegaConf.to_container(self.config.flyconfig.logging))
            logger.info("FlyLogger is initialized!")

            if self.chdir:
                logger.info(f"Working directory is changed to {working_dir_path}")
        elif self.rank != 0 and self.logging:
            # for other ranks, we initialize a debug level logger
            logging.basicConfig(format=f'[%(asctime)s][%(name)s][%(levelname)s][RANK {self.rank}] - %(message)s', level=logging.DEBUG)

        # save the current configuration, only rank 0 can save
        if self.rank == 0:
            # save the entire config directory
            cwd = self.config.flyconfig.runtime.cwd
            config_path = self.config.flyconfig.runtime.config_path
            cwd_config_dirpath = os.path.join(cwd, os.path.dirname(config_path))
            save_config_path = os.path.join(self.config.flyconfig.output_subdir, "saved_config")

            if not os.path.exists(save_config_path):
                shutil.copytree(cwd_config_dirpath, save_config_path)

                final_config_path = os.path.join(self.config.flyconfig.output_subdir, "config.yml")

                with open(final_config_path, "w") as f:
                    OmegaConf.save(self.config, f)

        self.initialized = True

    def is_initialized(self) -> bool:
        return self.initialized

    def clear(self) -> None:
        self.initialized = False
        os.chdir(self.config.flyconfig.runtime.cwd)