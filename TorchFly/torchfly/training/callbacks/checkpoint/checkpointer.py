import os
import glob
import time
import datetime
import torch
import pickle
import logging
import torchfly
from typing import Any, List, Dict, Iterator, Tuple

logger = logging.getLogger(__name__)


class Checkpointer:
    """
    Attributes:
        num_checkpoints_to_keep: Total number of checkpoints to keep
        keep_checkpoint_every_num_seconds: Keep checkpoints every x number of seconds without removing them
        storage_dir: Location to store the checkpoints
    """
    def __init__(
        self,
        sync_every_save: bool = True,
        async_save=False,
        num_checkpoints_to_keep: int = 1000,
        keep_checkpoint_every_num_seconds: float = 3600,
        storage_dir: str = "Checkpoints"
    ):
        self.sync_every_save = sync_every_save
        self.async_save = async_save
        self.num_checkpoints_to_keep = num_checkpoints_to_keep
        self.keep_checkpoint_every_num_seconds = keep_checkpoint_every_num_seconds
        self.storage_dir = storage_dir
        self._saved_checkpoint_paths: List[Tuple[float, str]] = []
        self._last_checkpoint_time = datetime.datetime.now()
        self.background_tasks = []

        os.makedirs(storage_dir, exist_ok=True)

    def save_checkpoint(self, stamp: str, model_state_dict: Dict[str, Any], trainer_state_dict: Dict[str, Any]) -> None:
        """
        Args:
            stamp: A string to identify the checkpoint. It can just be the epoch number
            states: A dictionary to store all necessary information for later restoring
        """
        # synchronize background tasks
        if self.sync_every_save and self.async_save:
            for process in self.background_tasks:
                torchfly.async_wait(process)
                logger.debug("Waiting for history job to finish!")
            self.background_tasks = []

        model_state_path = os.path.join(self.storage_dir, f"{stamp}_model_state.pth")
        trainer_state_path = os.path.join(self.storage_dir, f"{stamp}_trainer_state.pth")

        # remove the old one
        if self.num_checkpoints_to_keep >= 0:
            self._saved_checkpoint_paths.append((datetime.datetime.now(), model_state_path, trainer_state_path))
            trainer_state_dict["checkpointer_state_dict"] = self.state_dict()

            # save the states
            if self.async_save:
                process1 = torchfly.async_save(model_state_dict, model_state_path)
                process2 = torchfly.async_save(trainer_state_dict, trainer_state_path)
                self.background_tasks.append(process1)
                self.background_tasks.append(process2)
            else:
                torch.save(model_state_dict, model_state_path)
                torch.save(trainer_state_dict, trainer_state_path)

            if len(self._saved_checkpoint_paths) > self.num_checkpoints_to_keep:
                for _ in range(len(self._saved_checkpoint_paths) - self.num_checkpoints_to_keep):
                    path_to_remove = self._saved_checkpoint_paths.pop(0)

                    # check time requirement
                    remove_path = True
                    if self.keep_checkpoint_every_num_seconds is not None:
                        save_time = path_to_remove[0]
                        time_since_checkpoint_kept = (save_time - self._last_checkpoint_time).total_seconds()
                        if time_since_checkpoint_kept > self.keep_checkpoint_every_num_seconds:
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_checkpoint_time = save_time

                    if remove_path:
                        for fname in path_to_remove[1:]:
                            if os.path.isfile(fname):
                                logger.debug(f"Removing {fname}!")
                                os.remove(fname)

    def restore_latest_checkpoint(self) -> [Dict, None]:
        """
        DEPRECATED
        Returns:
            state_dict: return the checkpoint's state dict. None if there is nothing.
        """
        logger.warning("This function is deprecated!")
        files = glob.glob(os.path.join(self.storage_dir, "*model_state.pth"))
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

    def state_dict(self):
        states = {
            "_saved_checkpoint_paths":
                [
                    (str(saved_time), model_path, trainer_path)
                    for saved_time, model_path, trainer_path in self._saved_checkpoint_paths
                ],
            "_last_checkpoint_time": str(self._last_checkpoint_time)
        }
        return states

    def load_state_dict(self, states: Dict[str, Any]):
        self._saved_checkpoint_paths = [
            (datetime.datetime.strptime(saved_time, '%Y-%m-%d %H:%M:%S.%f'), model_path, trainer_path)
            for saved_time, model_path, trainer_path in states["_saved_checkpoint_paths"]
        ]
        self._last_checkpoint_time = datetime.datetime.strptime(states["_last_checkpoint_time"], '%Y-%m-%d %H:%M:%S.%f')