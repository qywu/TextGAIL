"""
This file is DEPRECATED!
"""

from typing import Any, Dict
import os
import sys
import time
import tqdm
import datetime
import collections
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from colorlog import colorlog
import atexit

from .events import Events
from .callback import Callback, handle_event
import logging

logger = logging.getLogger("torchfly.training.logger")
Trainer = Any


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook, Spyder or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


IN_NOTEBOOK = isnotebook()

if IN_NOTEBOOK:
    try:
        from IPython.display import clear_output, display, HTML
        import matplotlib.pyplot as plt
    except:
        logger.warn("Couldn't import ipywidgets properly, progress bar will use console behavior")
        IN_NOTEBOOK = False


def get_rank():
    """
    We use environment variables to pass the rank info
    Returns:
        rank: rank in the multi-node system 
        local_rank: local rank on a node
    """
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        rank = 0
        local_rank = 0
        os.environ["RANK"] = str(0)
        os.environ["LOCAL_RANK"] = str(0)
    else:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

    return rank, local_rank


@Callback.register("log_handler")
class LogHandler(Callback):
    """
    Callback that handles all Tensorboard logging.
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.rank, _ = get_rank()
        self.cumulative_time = 0.0

        self.history_log_dict = {}
        self.smooth_coef = 0.95

        # Log in seconds or steps
        if config.training.logging.steps_interval > 0:
            self.log_in_seconds = False
        else:
            if config.training.logging.seconds_interval < 0:
                raise NotImplementedError("config.training.logging.seconds_interval must be larger than 0")
            self.log_in_seconds = True

        # Train in epochs or steps
        if config.training.total_num.epochs > 0:
            self.training_in_epoch = True
        else:
            if config.training.total_num.update_steps < 0:
                raise NotImplementedError("config.training.total_num.updated_steps must be larger than 0")
            self.training_in_epoch = False

        if self.rank > 0:
            logging.shutdown()
            logger.handlers.clear()

        # correctly handles exception
        atexit.register(self.__del__)

    @handle_event(Events.INITIALIZE, priority=100)
    def report_init_config(self, trainer: Trainer):
        logger.info(OmegaConf.to_yaml(self.config))

    # @handle_event(Events.TRAIN_BEGIN, priority=195)
    # def setup_logging(self, trainer: Trainer):
    #     if IN_NOTEBOOK:
    #         trainer.train_dataloader = tqdm.tqdm(trainer.train_dataloader, total=trainer.total_num_training_steps)

    # Setup timing
    @handle_event(Events.TRAIN_BEGIN, priority=155)
    def setup_timer(self, trainer: Trainer):
        if self.rank == 0:
            self.last_log_time = time.time()
            self.epoch_start_time = time.time()
            # Info the start
            logger.info("Training Starts!")

            if trainer.global_step_count > 0:
                self.resume_training = True
                logger.info("Resume the training!")
            else:
                self.resume_training = False

        self.last_log_global_step = 0

    @handle_event(Events.TRAIN_BEGIN, priority=140)
    def setup_tensorboard(self, trainer: Trainer):
        # Setup tensorboard
        if self.rank == 0:
            log_dir = os.path.join(os.getcwd(), "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            self.tensorboard = SummaryWriter(log_dir=log_dir, purge_step=trainer.global_step_count)

        if not self.training_in_epoch:
            logger.info(f"Training total num of steps: {trainer.total_num_update_steps}")

    @handle_event(Events.EPOCH_BEGIN)
    def setup_epoch_timer(self, trainer: Trainer):
        if self.rank == 0:
            if self.training_in_epoch:
                logger.info("Epoch %d/%d", trainer.epochs_trained + 1, trainer.total_num_epochs)
                self.epoch_start_time = time.time()
            else:
                logger.info(f"Epoch {trainer.epochs_trained + 1}")

    @handle_event(Events.BATCH_END)
    def on_batch_end(self, trainer: Trainer):
        if self.rank == 0:
            if self.resume_training:
                self.log(trainer, trainer.tmp_vars["log_dict"])
                self.resume_training = False
            elif self.log_in_seconds:
                current_time = time.time()
                iter_elapsed_time = current_time - self.last_log_time

                if iter_elapsed_time > self.config.training.logging.seconds_interval:
                    self.log(trainer, trainer.tmp_vars["log_dict"])
            else:
                if (trainer.global_step_count + 1) % self.config.training.logging.steps_interval == 0:
                    self.log(trainer, trainer.tmp_vars["log_dict"])

    @handle_event(Events.EPOCH_END, priority=100)
    def on_epoch_end(self, trainer: Trainer):
        if self.rank == 0:
            if self.training_in_epoch:
                epoch_elapsed_time = time.time() - self.epoch_start_time
                logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

    @handle_event(Events.TRAIN_END)
    def on_train_end(self, trainer: Trainer):
        if self.rank == 0:
            logging.shutdown()
            self.tensorboard.close()
            logger.info("Training Finishes!")

    @handle_event(Events.VALIDATE_BEGIN)
    def info_valid_begin(self, trainer: Trainer):
        if self.rank == 0 and trainer.validation_dataloader is not None:
            updated_steps = trainer.global_step_count // self.config.training.optimization.gradient_accumulation_steps
            logger.info(f"Steps {updated_steps}: Validation Begins:")

    @handle_event(Events.VALIDATE_END)
    def show_valid_metrics(self, trainer: Trainer):
        if self.rank == 0 and trainer.validation_dataloader is not None:
            updated_steps = trainer.global_step_count // self.config.training.optimization.gradient_accumulation_steps

            if len(trainer.tmp_vars["validate_metrics"].items()) == 0:
                logger.warn(f"No metrics to report! Check if `get_metrics` is implemented.")

            for metric_name, value in trainer.tmp_vars["validate_metrics"].items():
                metric_name = metric_name[0].upper() + metric_name[1:]
                if not self.training_in_epoch:
                    logger.info(f"Steps {updated_steps}: Validation {metric_name} {value:4.4f}")
                else:
                    logger.info(f"Epoch {trainer.epochs_trained + 1}: Validation {metric_name} {value:4.4f}")

                # tensorboard
                if isinstance(value, float):
                    self.tensorboard.add_scalar("validate/" + metric_name, value, global_step=trainer.global_step_count)

    @handle_event(Events.TEST_BEGIN)
    def info_test_begin(self, trainer: Trainer):
        if self.rank == 0 and trainer.test_dataloader is not None:
            updated_steps = trainer.global_step_count // self.config.training.optimization.gradient_accumulation_steps
            logger.info(f"Steps {updated_steps}: Test Begins:")

    @handle_event(Events.TEST_END)
    def show_test_metrics(self, trainer: Trainer):
        if self.rank == 0 and trainer.test_dataloader is not None:
            updated_steps = trainer.global_step_count // self.config.training.optimization.gradient_accumulation_steps

            if len(trainer.tmp_vars["test_metrics"].items()) == 0:
                logger.warn(f"No metrics to report! Check if `get_metrics` is implemented.")

            for metric_name, value in trainer.tmp_vars["test_metrics"].items():
                metric_name = metric_name[0].upper() + metric_name[1:]
                if not self.training_in_epoch:
                    logger.info(f"Steps {updated_steps}: Test {metric_name} {value:4.4f}")
                else:
                    logger.info(f"Epoch {trainer.epochs_trained + 1}: Test {metric_name} {value:4.4f}")

                # tensorboard
                if isinstance(value, float):
                    self.tensorboard.add_scalar("test/" + metric_name, value, global_step=trainer.global_step_count)

    def log(self, trainer: Trainer, log_dict: Dict[str, float]):
        """
        Args:
            trainer: Trainer class
            log_dict: Dict
        """
        updated_steps = trainer.global_step_count // self.config.training.optimization.gradient_accumulation_steps

        if not self.training_in_epoch:
            percent = 100. * updated_steps / trainer.epoch_num_update_steps
        else:
            percent = 100. * trainer.local_step_count / trainer.epoch_num_update_steps

        iter_elapsed_time = time.time() - self.last_log_time
        elapsed_steps = trainer.global_step_count - self.last_log_global_step

        speed = elapsed_steps * self.config.training.batch_size * trainer.world_size / iter_elapsed_time
        self.cumulative_time += iter_elapsed_time

        if not self.training_in_epoch:
            log_string = (f"Train Steps - {updated_steps:<6} - " f"[{percent:7.4f}%] - " f"Speed: {speed:4.1f} - ")
        else:
            log_string = (
                f"Train Epoch: [{trainer.epochs_trained + 1}/{self.config.training.total_num.epochs}] - "
                f"Steps: {updated_steps:<6} - "
                f"[{percent:7.4f}%] - "
                f"Speed: {speed:4.1f} - "
            )

        # Smooth values in the log_dict
        for key, value in log_dict.items():
            if key in self.history_log_dict:
                if not isinstance(self.history_log_dict[key], float):
                    logger.error(f"{key} is not a float")
                    raise NotImplementedError(f"{key} is not a float")
                self.history_log_dict[key
                                     ] = self.history_log_dict[key] * self.smooth_coef + value * (1 - self.smooth_coef)
            else:
                self.history_log_dict[key] = value

            self.tensorboard.add_scalar(f"train/{key.lower()}", value, trainer.global_step_count + 1)

        # Add to Logging
        for key, value in self.history_log_dict.items():
            if key[0] != '_':
                log_string = log_string + f"{key[0].upper() + key[1:]}: {value:7.4f} - "

        log_string = log_string.strip(" - ")

        # Only log when not in notebook
        if not IN_NOTEBOOK:
            logger.info(log_string)
        else:
            trainer.train_dataloader.set_postfix(**log_dict)

        self.tensorboard.add_scalar("train/speed", speed, trainer.global_step_count + 1)
        self.tensorboard.add_scalar("train/cumulative_time", self.cumulative_time, trainer.global_step_count + 1)

        self.last_log_time = time.time()
        self.last_log_global_step = trainer.global_step_count

    def __del__(self):
        if self.rank == 0:
            logging.shutdown()
            logger.handlers.clear()

            if hasattr(self, "tensorboard"):
                self.tensorboard.close()

    def state_dict(self):
        state_dict = {"cumulative_time": self.cumulative_time, "history_log_dict": self.history_log_dict}
        return state_dict

    def load_state_dict(self, state_dict):
        self.cumulative_time = state_dict["cumulative_time"]
        self.history_log_dict = state_dict["history_log_dict"]