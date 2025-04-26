import os
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader

def get_tokenized_dataset(tokenizer, cache_dir: str, num_labels: int, max_length: int = 1024):
    """Load & cache a tokenized GoEmotions DatasetDict with train/validation/test."""
    if os.path.isdir(cache_dir):
        ds = load_from_disk(cache_dir)
    else:
        raw = load_dataset("go_emotions")  # splits: train/validation/test
        def preprocess(batch):
            toks = tokenizer(batch["text"],
                             truncation=True,
                             max_length=max_length,
                             padding=False)
            mh = np.zeros((len(batch["labels"]), num_labels), dtype=np.float32)
            for i, labs in enumerate(batch["labels"]):
                mh[i, labs] = 1.0
            toks["label"] = mh.tolist()
            return toks

        ds = raw.map(preprocess, batched=True, remove_columns=["text"])
        for split in ds:
            ds[split].set_format(type="torch",
                                 columns=["input_ids","attention_mask","label"])
        ds.save_to_disk(cache_dir)
    return ds

class CustomCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, batch):
        batch = self.base_collator(batch)
        if 'labels' in batch:
            batch['labels'] = batch['labels'].float()
        return batch


def get_dataloader(dataset: torch.utils.data.Dataset,
                   tokenizer,
                   batch_size: int):
    """Wrap a dataset into a DataLoader with padding collator."""
    collator = CustomCollator(tokenizer)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

