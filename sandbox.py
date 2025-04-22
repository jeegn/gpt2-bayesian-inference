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
)

from peft import LoraConfig, get_peft_model # noqa: E402
from datasets import Dataset, load_dataset# noqa: E402


torch.manual_seed(8)
np.random.seed(8)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "tingtone/go_emo_gpt"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = load_dataset("go_emotions")
test_dataset = dataset["test"]
train_dataset = dataset["train"]
# val_dataset = dataset["validation"]
num_labels = dataset["train"].features["labels"].feature.num_classes  # => 28
print(f"Number of labels: {num_labels}")

train_dataset = train_dataset.shuffle(seed=42).select(range(10))
def preprocess(batch):
    # 1) Tokenize to lists only
    toks = tokenizer(
        batch["text"],
        truncation=True,      # truncate long sequences
        max_length=1024,      # but DON'T pad here
        padding=False
    )
    # 2) Build multi-hot labels as Python lists
    mh = np.zeros((len(batch["labels"]), num_labels), dtype=np.float32)
    for i, labs in enumerate(batch["labels"]):
        mh[i, labs] = 1
    toks["label"] = mh.tolist()
    return toks

# Apply without setting torch format
train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["text"])
# val_dataset   = val_dataset.map(preprocess, batched=True, remove_columns=["text"])
test_dataset  = test_dataset.map(preprocess, batched=True, remove_columns=["text"])

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
# val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

collator = DataCollatorWithPadding(tokenizer)


train_dataloader = data_utils.DataLoader(
    train_dataset, batch_size=100, collate_fn=collator
)

# val_dataloader = data_utils.DataLoader(
#     val_dataset, batch_size=100, collate_fn=collator
# )
test_dataloader = data_utils.DataLoader(
    test_dataset, batch_size=100, collate_fn=collator
)

# Define the model
model = MyGPT2(
    tokenizer=tokenizer,
    model_name=model_name,
    num_labels=num_labels
)

model.eval()

la = Laplace(
    model,
    likelihood="classification",
    subset_of_weights="last_layer",
    hessian_structure="full",
    # This must reflect faithfully the reduction technique used in the model
    # Otherwise, correctness is not guaranteed
    feature_reduction="pick_last",
)
la.fit(train_loader=train_dataloader)
la.optimize_prior_precision()



