from typing import Dict
import math
import torch
import torch.nn as nn
from transformers import RobertaTokenizer

import torchfly
from torchfly.nn.transformers import GPT2LMHeadModel
from torchfly.training import FlyModel
from torchfly.metrics import CategoricalAccuracy, Average, MovingAverage, Speed
from torchfly.common.download import get_pretrained_weights

# pylint: disable=no-member


class Seq2Seq(FlyModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = GPT2LMHeadModel(config.model)
        self.decoder = self.encoder  # shared encoder-decoder
        self.pad_token_id = config.model.pad_token_id
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.model.pad_token_id)

        # load pretrained weights
        model_weights = get_pretrained_weights("roberta-tokenized-gpt2")
        print(self.encoder.load_state_dict(model_weights, strict=False))

    def configure_metrics(self):
        self.training_metrics = {"loss": MovingAverage()}
        self.evaluation_metrics = {"loss": Average()}

    def forward(self, batch):
        batch["source_mask"] = batch["source_token_ids"] != self.pad_token_id
        batch["target_mask"] = batch["target_token_ids"] != self.pad_token_id
        batch["source_position_ids"] = batch["source_mask"].cumsum(-1) - 1
        batch["target_position_ids"] = batch["source_position_ids"][:,
                                                                    -1].unsqueeze(-1) + batch["target_mask"].cumsum(-1)

        # encoder part
        _, past = self.encoder(
            input_ids=batch["source_token_ids"],
            position_ids=batch['source_position_ids'],
            attention_mask=batch["source_mask"],
            past=None
        )

        joint_mask = torch.cat([batch["source_mask"], batch["target_mask"]], dim=1)

        logits, _ = self.decoder(
            input_ids=batch["target_token_ids"],
            position_ids=batch['target_position_ids'],
            attention_mask=joint_mask,
            past=past
        )

        loss = self.compute_lm_loss(batch["target_token_ids"], logits, batch["target_mask"])
        results = {"loss": loss}
        # record training statistics
        self.training_metrics["loss"](loss.item())
        return results

    def compute_lm_loss(self, input_ids, logits, mask):
        logits = logits[:, :-1].contiguous()
        target = input_ids[:, 1:].contiguous()
        #mask = mask[:, 1:].float()
        #return self.criterion(logits, target, mask)
        return self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))

    def predict(self, batch):
        results = self.forward(batch)
        # record evaluation statistics
        self.evaluation_metrics["loss"](results["loss"].item())

    def get_training_metrics(self) -> Dict[str, str]:
        loss = self.training_metrics["loss"].get_metric()
        metrics = {"loss": (f"{loss:.4f}", loss)}
        return metrics

    def get_evaluation_metrics(self, reset: bool = False) -> Dict[str, float]:
        loss = self.evaluation_metrics["loss"].get_metric()
        ppl = math.exp(loss)
        metrics = {"loss": (f"{loss:.4f}", loss), "ppl": (f"{ppl:.4f}", ppl)}
        return metrics