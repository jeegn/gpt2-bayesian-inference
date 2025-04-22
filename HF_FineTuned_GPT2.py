from collections.abc import MutableMapping
import torch
from torch import nn
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    PreTrainedTokenizer,
)

class MyGPT2(nn.Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model_name, num_labels) -> None:
        super().__init__()
        config = GPT2Config.from_pretrained(model_name)
        config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = num_labels
        self.hf_model = GPT2ForSequenceClassification.from_pretrained(
            model_name, config=config
        )

    def forward(self, data: MutableMapping) -> torch.Tensor:
        """
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, n_classes)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        attn_mask = data["attention_mask"].to(device)
        output_dict = self.hf_model(input_ids=input_ids, attention_mask=attn_mask)
        return output_dict.logits

