from collections.abc import MutableMapping
import torch
from torch import nn
from transformers import PreTrainedTokenizer, GPT2ForSequenceClassification

class MyGPT2(nn.Module):
    """
    Huggingface GPT‑2 wrapper that accepts an already‑loaded
    GPT2ForSequenceClassification model.

    Args:
        tokenizer: The tokenizer used for preprocessing (for pad_token_id).
        model:     A GPT2ForSequenceClassification instance,
                   e.g. loaded with local_files_only=True.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: GPT2ForSequenceClassification
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.hf_model  = model

        # Make sure the model knows about the pad token
        self.hf_model.config.pad_token_id = tokenizer.pad_token_id

    def forward(self, data: MutableMapping) -> torch.Tensor:
        """
        Args:
          data: a dict with at least "input_ids" and "attention_mask" tensors.

        Returns:
          logits: Tensor of shape (batch_size, num_labels).
        """
        device = next(self.parameters()).device
        input_ids   = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)

        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return outputs.logits