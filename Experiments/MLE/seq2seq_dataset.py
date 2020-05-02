import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class Seq2SeqDataset(Dataset):
    """
    A Simple Seq2Seq Dataset Implementation
    """
    def __init__(self, filename, tokenizer, add_bos_token=True, add_eos_token=True):
        with open(filename, "r") as f:
            self.data = f.readlines()

        self.data = [json.loads(item) for item in self.data]
        self.tokenizer = tokenizer
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def __getitem__(self, index):
        item = self.data[index]
        source_token_ids = self.tokenizer.encode(item["source"], add_special_tokens=False)
        target_token_ids = self.tokenizer.encode(item["target"], add_special_tokens=False)

        if self.add_bos_token:
            target_token_ids.insert(0, self.tokenizer.bos_token_id)

        if self.add_eos_token:
            target_token_ids.append(self.tokenizer.eos_token_id)

        item["source_token_ids"] = torch.LongTensor(source_token_ids)
        item["target_token_ids"] = torch.LongTensor(target_token_ids)
        return item

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        new_batch = {}
        new_batch["source_token_ids"] = pad_sequence(
            [item["source_token_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        new_batch["target_token_ids"] = pad_sequence(
            [item["target_token_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return new_batch


