import os
import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader

def get_goEmo_dataset(tokenizer, cache_dir: str, num_labels: int, max_length: int = 1024):
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

def get_emoint_dataset(tokenizer, cache_dir: str, max_length: int = 256):
    """Load & cache the EmoInt OOD dataset with one-hot 4-dim labels."""
    if os.path.isdir(cache_dir):
        print(f"Loading EmoInt OOD dataset from cache: {cache_dir}")
        dataset = load_from_disk(cache_dir)
        return dataset

    print("Downloading and tokenizing EmoInt OOD dataset...")
    ood_emotions = ["sadness", "joy", "fear", "anger"]
    ood2id = {k: v for v, k in enumerate(ood_emotions)}
    BASE_URL = "https://raw.githubusercontent.com/SEERNET/EmoInt/refs/heads/master/emoint/resources/emoint"

    df = pd.concat([
        pd.read_csv(f"{BASE_URL}/{emo}-ratings-0to1.train.txt", sep="\t", names=["index", "text", "text_label", "intensity"])
        for emo in ood_emotions
    ])
    df = df.drop(columns=["index", "intensity"]).reset_index(drop=True)
    df['label_id'] = df['text_label'].map(ood2id)   # mapped to ints 0-3

    # --- One-hot encode labels ---
    num_classes = len(ood_emotions)
    labels = []
    for label in df["label_id"]:
        one_hot = [0] * num_classes
        one_hot[label] = 1
        labels.append(one_hot)

    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    data = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "label": labels,  # now a list of one-hot vectors
    }

    dataset = HFDataset.from_dict(data)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(cache_dir)

    print(f"Saved EmoInt OOD dataset to cache: {cache_dir}")
    return dataset

def map_preds_to_ood(preds: torch.Tensor) -> torch.Tensor:
    """
    Map the predictions from the model to the OOD labels.
    The mapping is as follows:
    0 -> sadness
    1 -> joy
    2 -> fear
    3 -> anger
    """
    high2fine = {
    "anger":   ["anger", "annoyance", "disapproval"],
    "fear":    ["fear", "nervousness"],
    "joy":     ["joy", "amusement", "approval", "excitement",
                "gratitude", "love", "optimism", "relief",
                "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment",
                "grief", "remorse"],}
    
    id_emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    ood_emotions = ["sadness", "joy", "fear", "anger"]

    emotion2id = {e: i for i, e in enumerate(id_emotions)}

    B = preds.shape[0]
    ood_preds = np.zeros((B, len(ood_emotions)), dtype=int)

    # Fine emotion → index lookup
    emotion2id = {e: i for i, e in enumerate(id_emotions)}

    for idx, coarse_emotion in enumerate(ood_emotions):
        fine_emotions = high2fine[coarse_emotion]
        fine_ids = [emotion2id[e] for e in fine_emotions if e in emotion2id]
        if fine_ids:
            ood_preds[:, idx] = np.any(preds[:, fine_ids], axis=1).astype(int)

    return ood_preds

def map_probs_to_ood(probs: np.ndarray) -> np.ndarray:
    """
    Maps (B, 28) predicted probabilities to (B, 4) coarse-grained OOD probabilities.
    """
    high2fine = {
    "anger":   ["anger", "annoyance", "disapproval"],
    "fear":    ["fear", "nervousness"],
    "joy":     ["joy", "amusement", "approval", "excitement",
                "gratitude", "love", "optimism", "relief",
                "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment",
                "grief", "remorse"],}
    
    id_emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    ood_emotions = ["sadness", "joy", "fear", "anger"]
    B = probs.shape[0]
    ood_probs = np.zeros((B, len(ood_emotions)), dtype=np.float32)

    # Fine emotion → index lookup
    emotion2id = {e: i for i, e in enumerate(id_emotions)}

    for idx, coarse_emotion in enumerate(ood_emotions):
        fine_emotions = high2fine[coarse_emotion]
        fine_ids = [emotion2id[e] for e in fine_emotions if e in emotion2id]
        if fine_ids:
            # Smart: take maximum probability among fine-grained emotions
            ood_probs[:, idx] = np.max(probs[:, fine_ids], axis=1)

    return ood_probs