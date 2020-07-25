import hydra
import hydra.experimental
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
from omegaconf import DictConfig

from torchfly.text.decode import TransformerDecoder
from torchfly.common import set_random_seed
from torchfly.text.rl import TextRLRewardFunc

from configure_dataloader import DataLoaderHandler, TextRLCollator

from model import TextGAILModel
from textgail_discriminator import TextGAILDiscriminator
from textgail_trainerloop import TextGAILTrainerLoop 

import logging


@hydra.main(config_path="config/config.yaml", strict=False)
def main(config=None):
    set_random_seed(config.training.random_seed)
    dataloader_handler = DataLoaderHandler(config)

    model = TextGAILModel(config)
    model_weights = torch.load(config.task.weights_path)
    model.generator.load_state_dict(model_weights, strict=False)
    model = model.cuda()

    # Register your transformer for decoding
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    decoder_helper = TransformerDecoder(config.decode)
    decoder_helper.register_generator(model.generator.decoder)
    decoder_helper.register_tokenizer(tokenizer)
    decoder_helper.prepare_model_inputs_for_generation = model.generator.prepare_model_inputs_for_generation

    reward_func = TextGAILDiscriminator(config, tokenizer, model.discriminator)

    trainer = TextGAILTrainerLoop(config=config,
                                reward_func=reward_func, 
                                decoder_helper=decoder_helper,
                                model=model, 
                                train_dataloader_fn=dataloader_handler.train_dataloader,
                                valid_dataloader_fn=dataloader_handler.valid_dataloader)
                            
    trainer.train()

if __name__ == "__main__":
    main()