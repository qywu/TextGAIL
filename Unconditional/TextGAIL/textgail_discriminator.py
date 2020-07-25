import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchfly.text.rl import TextRLRewardFunc

import logging

class TextGAILDiscriminator(TextRLRewardFunc):
    def __init__(self, config, tokenizer, discriminator):
        self.config = config
        self.discriminator = discriminator
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = torch.LongTensor([self.tokenizer.sep_token_id])

    def get_reward(self, states, actions):
        device = next(self.discriminator.parameters()).device
        inputs = []
        for i in range(len(states)):
            real_sample = states[i]['target_token_ids']        
            generated_sample = actions["tokens"][i]
            inputs.append(real_sample)
            inputs.append(generated_sample)
            
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)
        input_ids = input_ids.view(-1, 2, input_ids.shape[-1])
            
        batch = {"input_ids": input_ids.to(device)}
        results = self.discriminator(batch)
        return results["reward"].detach().cpu().numpy()

    def get_loss(self, states, actions):
        device = next(self.discriminator.parameters()).device
        inputs = []
        for i in range(len(states)):
            real_sample = states[i]['target_token_ids']
            generated_sample = actions[i]
            inputs.append(real_sample)
            inputs.append(generated_sample)

        input_ids = pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)
        input_ids = input_ids.view(-1, 2, input_ids.shape[-1])
            
        batch = {"input_ids": input_ids.to(device)}
        results = self.discriminator(batch)
        return results