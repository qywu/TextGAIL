from typing import Any, Dict
import os
import sys
import time
import glob
import torch
import pickle
import logging
from omegaconf import DictConfig
# from apex import amp

from ..events import Events
from ..callback import Callback, handle_event
from ....utils.distributed import get_rank

logger = logging.getLogger("resumer")

__all__ = ["Resume"]
Trainer = Any


@Callback.register("resume")
class Resume(Callback):
    """
    Callback that resumes the training from a checkpoint
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.storage_dir = self.config.training.checkpointing.directory
        self.restored_states = None

    @handle_event(Events.INITIALIZE, priority=199)
    def setup(self, trainer: Trainer):
        # Search for the latest checkpoint
        if self.config.training.resume.resume:
            logger.info("Try to restore the latest checkpoint")
            self.restored_states = self.restore_latest_checkpoint(self.storage_dir)

            # Check if restore states is valid
            if self.restored_states:
                file_path = self.restored_states[2]
                print(f"Found checkpoint at {file_path}!")
            else:
                logger.warn("Fail to find any checkpoint! Start new training!")
        else:
            self.restored_states = None

    @handle_event(Events.TRAIN_BEGIN, priority=170)
    def load_checkpoint(self, trainer: Trainer):
        # Resume the training
        if self.restored_states and self.config.training.resume.resume:
            # Load Model State
            if self.config.training.resume.resume_model:
                trainer.set_model_state(self.restored_states[0])
            # Load Everything Else
            trainer.set_trainer_state(self.restored_states[1])

            file_path = self.restored_states[2]
            print(f"RANK {get_rank()} has loaded checkpoint at {file_path}!")

    def restore_latest_checkpoint(self, dirpath) -> [Dict, None]:
        """
        Returns:
            state_dict: return the checkpoint's state dict. None if there is nothing.
        """
        files = glob.glob(os.path.join(dirpath, "*model_state.pth"))
        # sort files based on time
        sorted_files = sorted(files, key=os.path.getctime, reverse=True)

        for latest_file_path in sorted_files:
            trainer_state_file = latest_file_path.split("model_state.pth")[0] + "trainer_state.pth"
            model_state_file = latest_file_path

            try:
                model_state_dict = torch.load(model_state_file, map_location="cpu")
                trainer_state_dict = torch.load(trainer_state_file, map_location="cpu")

                trainer_state_dict["file_path"] = latest_file_path
                logger.info(f"Loading checkpoint {latest_file_path}")
                return (model_state_dict, trainer_state_dict, latest_file_path)
            except (pickle.UnpicklingError, RuntimeError, TypeError, FileNotFoundError):
                # skip and remove the corrupted files
                logger.info(f"Checkpoint {trainer_state_file} is corrupted. It will be deleted.")
                os.remove(trainer_state_file)
                os.remove(model_state_file)
                continue

        # if found files but failed to load them
        if len(files) > 0:
            logger.info("Fail to retrieve the old checkpoints!")
        # if there is nothing to restore, return None
        return None
