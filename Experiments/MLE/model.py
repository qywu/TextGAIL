from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer

import torchfly
from torchfly.nn.transformers import GPT2LMHeadModel
from torchfly.training import FlyModule
from torchfly.nn.losses import SequenceCrossEntropyLoss
from torchfly.metrics import Average
from torchfly.common.download import get_pretrained_weights

# pylint: disable=no-member


class Seq2Seq(FlyModule):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = GPT2LMHeadModel(config.model)
        self.decoder = self.encoder  # shared encoder-decoder
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        # self.criterion = SequenceCrossEntropyLoss(reduce="sentence")
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.model.pad_token_id)
        self._perplexity = Average()

        # load pretrained weights
        model_weights = get_pretrained_weights("roberta-tokenized-gpt2")
        print(self.encoder.load_state_dict(model_weights, strict=False))

    def forward(self, batch):
        batch["source_mask"] = batch["source_token_ids"] != self.tokenizer.pad_token_id
        batch["target_mask"] = batch["target_token_ids"] != self.tokenizer.pad_token_id
        batch["source_position_ids"] = batch["source_mask"].cumsum(-1) - 1
        batch["target_position_ids"] = batch["source_position_ids"][:, -1].unsqueeze(-1
                                                                                    ) + batch["target_mask"].cumsum(-1)

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
        return results

    def predict(self, batch):
        results = self.forward(batch)
        self._perplexity(results["loss"].exp().item())

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ppl = self._perplexity.get_metric(reset)
        metrics = {"perplexity": ppl}
        return metrics

    def compute_lm_loss(self, input_ids, logits, mask):
        logits = logits[:, :-1].contiguous()
        target = input_ids[:, 1:].contiguous()
        #mask = mask[:, 1:].float()
        #return self.criterion(logits, target, mask)
        return self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))