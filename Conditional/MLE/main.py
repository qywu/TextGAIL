import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from torchfly.flylogger import FlyLogger
from torchfly.flyconfig import FlyConfig
from torchfly.training import TrainerLoop
from torchfly.common.download import get_pretrained_weights
from torchfly.common import set_random_seed

from model import Seq2Seq
from configure_dataloader import DataLoaderHandler

import logging
logger = logging.getLogger(__name__)


def main():
    config = FlyConfig.load()
    fly_logger = FlyLogger(config)

    set_random_seed(config.training.random_seed)
    dataloader_handler = DataLoaderHandler(config)
    # define the model here
    model = Seq2Seq(config)

    trainer = TrainerLoop(config, model, dataloader_handler.train_dataloader, dataloader_handler.valid_dataloader, dataloader_handler.test_dataloader)

    trainer.train()


if __name__ == "__main__":
    main()