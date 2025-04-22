# Pretrained GPT-2 with a custom head
# This code defines a custom GPT-2 model with a multi-layer perceptron (MLP) head for classification tasks.
# It uses the Hugging Face Transformers library to build a model that can be fine-tuned for multi-label classification.
#

import torch
from torch import nn
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config

class GPT2WithCustomHead(GPT2PreTrainedModel):
    def __init__(
        self,
        config: GPT2Config,
        head_hidden_dims: list[int] = [512, 256],  # two hidden layers
        dropout: float = 0.1
    ):
        super().__init__(config)
        self.gpt2 = GPT2Model(config)

        # build an MLP with arbitrary depth
        layers = []
        in_dim = config.n_embd
        for h in head_hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        # final projection
        layers.append(nn.Linear(in_dim, config.num_labels))

        self.classifier = nn.Sequential(*layers)
        self.init_weights()  # important!

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        outputs = self.gpt2(input_ids=input_ids, 
                            attention_mask=attention_mask)
        # grab last token’s embedding (or pool differently)
        hidden = outputs.last_hidden_state[:, -1, :]  # [batch, n_embd]
        logits = self.classifier(hidden)              # [batch, num_labels]

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits}

# Usage
# base_config = GPT2Config.from_pretrained("gpt2", num_labels=28)
# model = GPT2WithCustomHead(
#     config=base_config,
#     head_hidden_dims=[1024, 512, 256],  # 3‑layer head
#     dropout=0.2
# )