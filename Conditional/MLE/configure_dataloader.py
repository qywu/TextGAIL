import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from seq2seq_dataset import Seq2SeqDataset

# pylint:disable=no-member


class DataLoaderHandler:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def train_dataloader(self):

        train_filename = f"../../../../data/{self.config.task.name}/val.jsonl"
        train_dataset = Seq2SeqDataset(train_filename, self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )
        return train_dataloader

    def valid_dataloader(self):
        valid_filename = f"../../../../data/{self.config.task.name}/val.jsonl"
        val_dataset = Seq2SeqDataset(valid_filename, self.tokenizer)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.config.training.batch_size, collate_fn=val_dataset.collate_fn
        )
        return val_dataloader

    def test_dataloader(self):
        test_filename = f"../../../../data/{self.config.task.name}/test.jsonl"
        test_dataset = Seq2SeqDataset(test_filename, self.tokenizer)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.config.training.batch_size, collate_fn=test_dataset.collate_fn
        )
        return test_dataloader