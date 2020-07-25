import hydra
import hydra.experimental
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from torchfly.training.trainer import TrainerLoop
from torchfly.common.download import get_pretrained_weights
from torchfly.common import set_random_seed

from model import LanguageModel
from configure_dataloader import DataLoaderHandler

import logging
logger = logging.getLogger(__name__)


@hydra.main(config_path="config/config.yaml", strict=False)
def main(config=None):
    set_random_seed(config.training.random_seed)
    dataloader_handler = DataLoaderHandler(config)
    model = LanguageModel(config)
    trainer = TrainerLoop(config, model, dataloader_handler.train_dataloader, dataloader_handler.valid_dataloader)
    trainer.train()


if __name__ == "__main__":
    main()