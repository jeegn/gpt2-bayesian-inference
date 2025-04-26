from collections.abc import MutableMapping
from collections import UserDict
import numpy as np
import torch
from torch import nn
import torch.utils.data as data_utils
from tqdm.auto import tqdm
from HF_FineTuned_GPT2 import MyGPT2
from laplace import Laplace
from metrics import (
    elementwise_accuracy,
    subset_accuracy,
    hamming_loss,
    f1_scores,
    roc_auc_scores,
    log_loss_multilabel,
    brier_score_multilabel,
    get_calibration,
)


import logging
import warnings

logging.basicConfig(level="ERROR")
warnings.filterwarnings("ignore")

from transformers import ( # noqa: E402
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)

from huggingface_hub import Repository # noqa: E402
# from peft import LoraConfig, get_peft_model # noqa: E402
from datasets import Dataset, load_dataset, load_from_disk # noqa: E402
import os
import shutil
import tempfile

torch.manual_seed(8)
np.random.seed(8)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("CUDA current device:", torch.cuda.current_device(),
      torch.cuda.get_device_name(torch.cuda.current_device()))
    
num_labels = 28
print(f"Number of labels: {num_labels}")

DATA_CACHE = "/scratch/scholar/jdani/project/data_cache"
MODEL_CACHE = "/scratch/scholar/jdani/project/scratch_model"

PREFERRED_SUBFOLDER = "ood-epochs-10-ense-5"
HUGGINGFACE_REPO = "sawlachintan/gpt2-goemotions-ft"

def clone_and_prepare_model():
    tmp_dir = tempfile.mkdtemp()
    print(f"Cloning {HUGGINGFACE_REPO} into temp dir {tmp_dir}")
    repo = Repository(local_dir=tmp_dir, clone_from=HUGGINGFACE_REPO, use_auth_token=True)

    src = os.path.join(tmp_dir, PREFERRED_SUBFOLDER)
    dst = MODEL_CACHE
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    shutil.rmtree(tmp_dir)
    print(f"Model copied to {dst}")

# Load the tokenizer and model
if os.path.isdir(MODEL_CACHE):
    print("Loading model from local cache...")
else:
    print("Local model cache not found. Downloading model...")
    clone_and_prepare_model()

tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE)

# config = AutoConfig.from_pretrained(MODEL_CACHE)
config = AutoConfig.from_pretrained(
    "gpt2",
    num_labels=28,
    finetuning_task="multi_label_classification",
    pad_token_id=tokenizer.eos_token_id
)

pretrained_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CACHE,
    config=config
)

# Set pad token
tokenizer.pad_token_id = tokenizer.eos_token_id
# exit(1)
if os.path.isdir(DATA_CACHE):
    # 1) Load the tokenized dataset from disk
    tokenized_ds = load_from_disk(DATA_CACHE)
    print("Loaded dataset from local cache")
else:
    # 2) Download the raw dataset and tokenize it once
    raw = load_dataset("go_emotions")  # train/validation/test splits

    def preprocess(batch):
        toks = tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )
        mh = np.zeros((len(batch["labels"]), num_labels), dtype=np.float32)
        for i, labs in enumerate(batch["labels"]):
            mh[i, labs] = 1.0
        toks["labels"] = mh.tolist()
        return toks

    # apply to *all* splits
    tokenized_ds = raw.map(
        preprocess,
        batched=True,
        remove_columns=["text"]
    )

    # fix up torch format once
    for split in tokenized_ds:
        tokenized_ds[split].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

    # 3) Persist it
    tokenized_ds.save_to_disk(DATA_CACHE)

test_dataset  = tokenized_ds["test"]
train_dataset = tokenized_ds["train"]
# Subsample just 10 samples from train_dataset
train_dataset = train_dataset.select(range(len(train_dataset)))
print(f"Number of samples in the train dataset: {len(train_dataset)}")
# val_dataset = dataset["validation"]

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

class CustomCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, batch):
        batch = self.base_collator(batch)
        if 'labels' in batch:
            batch['labels'] = batch['labels'].float()
        return batch
collator = CustomCollator(tokenizer)
# collator = DataCollatorWithPadding(tokenizer)


train_dataloader = data_utils.DataLoader(
    train_dataset, batch_size=32, collate_fn=collator
)

# val_dataloader = data_utils.DataLoader(
#     val_dataset, batch_size=100, collate_fn=collator
# )
test_dataloader = data_utils.DataLoader(
    test_dataset, batch_size=32, collate_fn=collator
)

# Define the model
model = MyGPT2(
    tokenizer=tokenizer,
    model=pretrained_model
)
model = model.to(device)
model.eval()
print("Model is on:", next(model.parameters()).device)

# assume `model` is your MyGPT2 wrapper, already moved to device
# print("=== Full model architecture ===")
# print(model.hf_model)
# for p in model.hf_model.parameters():
#     p.requires_grad = False

# for p in model.hf_model.score.parameters():
#     p.requires_grad = True

# for p in model.hf_model.transformer.ln_f.parameters():
#     p.requires_grad = True

# Count total vs trainable
total_params     = sum(p.numel() for p in model.hf_model.parameters())
trainable_params = sum(p.numel() for p in model.hf_model.parameters() if p.requires_grad)

print(f"Total params:     {total_params}")
print(f"Trainable params: {trainable_params}")
print(f"Trainable %:      {100*trainable_params/total_params:.2f}%")
# exit(1)
# exit(1)

# Options for hessian structure:
# 'diag', 'kron', 'full', 'lowrank', 'gp'
# Options for subset of weights:
# 'all', 'last_layer', 'subnetwork'
# n_subset = x limits using only x training examples

la = Laplace(
    model,
    likelihood="classification",
    subset_of_weights="last_layer",
    hessian_structure="kron",
    # This must reflect faithfully the reduction technique used in the model
    # Otherwise, correctness is not guaranteed
    feature_reduction="pick_last",

)
la.fit(train_loader=train_dataloader)
la.optimize_prior_precision()

print("Laplace model fitted")

all_logits = []
all_la_preds = []
all_labels = []
total_loss = 0.0
# device = "cuda" if torch.cuda.is_available() else "mps"
for batch in tqdm(test_dataloader, desc="Evaluating"):
    data = {k: v.to(device) for k, v in batch.items()}
    # input_ids      = batch["input_ids"].to(device)
    # attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"]
    labels = labels.float()
    with torch.no_grad():
        logits = model(data)
        # loss   = loss_fn(logits, labels)
    la_pred = la(batch, pred_type="nn", link_approx='mc', n_samples=2000)

    # total_loss += loss.item()
    all_logits.append(logits.cpu().numpy())
    all_la_preds.append(la_pred.cpu().numpy())
    all_labels.append(labels.cpu().numpy())

print("Evaluating done")
all_logits = np.vstack(all_logits)
all_labels = np.vstack(all_labels)
all_la_preds = np.vstack(all_la_preds)
probs      = torch.sigmoid(torch.tensor(all_logits)).numpy()
preds      = (probs >= 0.5).astype(int)
la_preds  = (all_la_preds >= 0.5).astype(int)
print(all_labels.shape, all_logits.shape, all_la_preds.shape)
print(probs.shape, preds.shape, la_preds.shape)

# ——— Base GPT‑2 classifier metrics ———
print("=== Base GPT‑2 classifier ===")
print(f"Element-wise Accuracy: {elementwise_accuracy(all_labels, preds):.4f}")
print(f"Subset (Exact) Acc    : {subset_accuracy(all_labels, preds):.4f}")
print(f"Hamming Loss          : {hamming_loss(all_labels, preds):.4f}")
print(f"F1 (micro)            : {f1_scores(all_labels, preds, average='micro'): .4f}")
print(f"F1 (macro)            : {f1_scores(all_labels, preds, average='macro'): .4f}")
print(f"ROC AUC (micro)       : {roc_auc_scores(all_labels, probs, average='micro'): .4f}")
print(f"ROC AUC (macro)       : {roc_auc_scores(all_labels, probs, average='macro'): .4f}")
print(f"Log Loss              : {log_loss_multilabel(all_labels, probs):.4f}")
print(f"Brier Score           : {brier_score_multilabel(all_labels, probs):.4f}")
ece_cls, ece_glob, mce_glob = get_calibration(probs, all_labels)
print(f"Global ECE            : {ece_glob:.4f}")
print(f"Global MCE            : {mce_glob:.4f}")
print(f"Per‑class ECE         : {ece_cls}")

# ——— Laplace‑augmented model metrics ———
print("\n=== Last‑layer Laplace ===")
print(f"Element-wise Accuracy: {elementwise_accuracy(all_labels, la_preds):.4f}")
print(f"Subset (Exact) Acc    : {subset_accuracy(all_labels, la_preds):.4f}")
print(f"Hamming Loss          : {hamming_loss(all_labels, la_preds):.4f}")
print(f"F1 (micro)            : {f1_scores(all_labels, la_preds, average='micro'): .4f}")
print(f"F1 (macro)            : {f1_scores(all_labels, la_preds, average='macro'): .4f}")
print(f"ROC AUC (micro)       : {roc_auc_scores(all_labels, all_la_preds, average='micro'): .4f}")
print(f"ROC AUC (macro)       : {roc_auc_scores(all_labels, all_la_preds, average='macro'): .4f}")
print(f"Log Loss              : {log_loss_multilabel(all_labels, all_la_preds):.4f}")
print(f"Brier Score           : {brier_score_multilabel(all_labels, all_la_preds):.4f}")
ece_cls_la, ece_glob_la, mce_glob_la = get_calibration(all_la_preds, all_labels)
print(f"Global ECE            : {ece_glob_la:.4f}")
print(f"Global MCE            : {mce_glob_la:.4f}")
print(f"Per‑class ECE         : {ece_cls_la}")


