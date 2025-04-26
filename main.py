import argparse
import copy
import inspect
import pathlib
from types import SimpleNamespace
from typing import Dict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import random
from torch.optim import AdamW
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer
)
from huggingface_hub import Repository
from sklearn.metrics import f1_score

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    # init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


@dataclass
class GPTConfig:
    block_size: int = 512 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v,dropout_p=0.0, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, attention_mask=None, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # x = tok_emb + pos_emb
        x = tok_emb + pos_emb
        # (optional) you could mask out pad tokens here:
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        # logits = self.lm_head(x) # (B, T, vocab_size)
        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # return logits, loss
        logits = self.lm_head(x)   # (B, T, vocab_size)

        return SimpleNamespace(
            last_hidden_state = x,
            logits            = logits
        )

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class GPT2EmoClassifier(nn.Module):
    def __init__(self, model=None, model_name: str = "gpt2",
                     num_classes: int = 28, 
                     dropout: float = 0.1,
                     pooling: str = "mean"):
        super().__init__()
        if model is None:
            self.gpt2 = GPT2Model.from_pretrained(model_name)
        else: 
            self.gpt2 = model
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2.config.n_embd, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )
        self.pooling = pooling

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1)

        if self.pooling == "last":
            # index of last real token in each example
            idx = attention_mask.sum(dim=1) - 1          # (B,)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

        elif self.pooling == "mean":
            # sum then divide by length
            summed = (hidden * mask).sum(dim=1)          # (B, H)
            lengths = mask.sum(dim=1)                    # (B, 1)
            pooled = summed / lengths.clamp(min=1e-9)    # (B, H)

        else:  # max‐pool
            
            neg_inf = torch.finfo(hidden.dtype).min
            hidden_masked = hidden.masked_fill(mask == 0, neg_inf)
            pooled, _ = hidden_masked.max(dim=1)

        logits = self.classifier(self.dropout(pooled))
        return logits
    

class EnsembleClassifier(nn.Module):
    def __init__(self, base_model: GPT2EmoClassifier, ensemble_size: int):
        super().__init__()
        # Create N deep-copies so each can learn its own weights
        self.models = nn.ModuleList([
            copy.deepcopy(base_model) for _ in range(ensemble_size)
        ])

    def forward(self, input_ids, attention_mask=None):
        # Collect logits from each sub-model
        logits_list = [m(input_ids, attention_mask) for m in self.models]  
        # logits_list: List of tensors, each (B, C)
        stacked = torch.stack(logits_list, dim=0)  # (ensemble_size, B, C)
        return stacked.mean(dim=0)                 # (B, C)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"



RNG_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 on the single-label GoEmotions dataset (28 classes) with optional Distributed Data-Parallel (DDP) support via torchrun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # these will be overridden by torchrun when launched with multiple processes
    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", 0), help="[INTERNAL] local rank passed by torchrun")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--no_ood",action="store_true")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--model_loc", type=str, default="./log/model_00000.pt")
    parser.add_argument("--output_dir", type=str, default="./gpt2_goemotions_ft")
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # speed for fixed shapes


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

class GoEmotions(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        self.texts = df["text"].tolist()
        self.labels = torch.tensor(df["labels"].values, dtype=torch.long)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }

URL_ROOT = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "refs/heads/master/goemotions/data"
)
go_emotions = pd.read_csv(f"{URL_ROOT}/emotions.txt", names=['emotions']).reset_index(names=['id'])
emotion2id = go_emotions.set_index('emotions')['id'].to_dict()
id2emotion = go_emotions.set_index('id')['emotions'].to_dict()

thr_df = pd.read_csv("thresholds.csv")   # columns: emotion, threshold
thresholds_map = dict(zip(thr_df["emotion"], thr_df["threshold"]))
thresholds_list = [ thresholds_map[id2emotion[i]] for i in range(len(id2emotion)) ]

def load_data(no_ood: bool = True) -> Dict[str, pd.DataFrame]:
    train_df = pd.read_csv(f"{URL_ROOT}/train.tsv", sep="\t", names=["text", "labels", "id"])
    train_df = train_df[train_df["labels"].notna()].reset_index(drop=True)
    train_df["labels"] = train_df["labels"].apply(
            lambda s: [int(lbl) for lbl in s.split(",")]
        )
    

    val_df = pd.read_csv(f"{URL_ROOT}/dev.tsv", sep="\t", names=["text", "labels", "id"])
    val_df = val_df[val_df["labels"].notna()].reset_index(drop=True)
    val_df["labels"] = val_df["labels"].apply(
            lambda s: [int(lbl) for lbl in s.split(",")]
        )
    ood_emotions = ["sadness","joy","fear","anger"]
    ood2id = {k:v for v,k in enumerate(ood_emotions)}
    if no_ood:
        test_df = pd.read_csv(f"{URL_ROOT}/test.tsv", sep="\t", names=["text", "labels", "id"])
        test_df = test_df[test_df["labels"].notna()].reset_index(drop=True)
        test_df["labels"] = test_df["labels"].apply(
            lambda s: [int(lbl) for lbl in s.split(",")]
            )
    else:
        BASE_URL = "https://raw.githubusercontent.com/SEERNET/EmoInt/refs/heads/master/emoint/resources/emoint"

        test_df = (pd.concat([pd.read_csv(f"{BASE_URL}/{x}-ratings-0to1.train.txt", 
                                        sep="\t", 
                                        names=["index","text","text_label","intensity"]) for x in ood_emotions])
                    .drop(columns=['index','intensity']))
        test_df['labels'] = test_df['text_label'].map(ood2id).apply(lambda x: [x])
        test_df = test_df.sample(frac=1, random_state=587)
    # train_df = train_df[train_df["labels"].str.isdigit()].reset_index(drop=True)
    # val_df = val_df[val_df["labels"].str.isdigit()].reset_index(drop=True)
    # test_df = val_df[val_df["labels"].str.isdigit()].reset_index(drop=True)

    # train_df["labels"] = train_df["labels"].astype(int)
    # val_df["labels"] = val_df["labels"].astype(int)
    # test_df["labels"] = test_df["labels"].astype(int)

    return {"train": train_df, "val": val_df, "test":test_df, 
            "ood_emotions": ood_emotions, "ood2id": ood2id,
            }



class GoEmotions(Dataset):
    """
    A Dataset that tokenizes texts and builds a multi-hot label vector.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        num_classes: int = 28,
    ):
        self.texts       = df["text"].tolist()
        self.label_lists = df["labels"].tolist()   # List[List[int]]
        if "text_labels" in df.columns:
            self.text_labels = df['text_labels'].tolist()
        else:
            self.text_labels = []
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text        = self.texts[idx]
        label_idxs  = self.label_lists[idx]

        # tokenize (returns a dict of tensors with shape [1, L])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # build a multi-hot label vector of size num_classes
        labels = torch.zeros(self.num_classes, dtype=torch.float)
        labels[label_idxs] = 1.0

        # squeeze away the batch-dim from tokenization
        item = { k: v.squeeze(0) for k, v in enc.items() }
        item["labels"] = labels
        item['text_labels'] = self.text_labels
        return item

def ood_helper(logits: torch.Tensor,
    text_labels,
    emotion2id: Dict[str, int] = emotion2id, thresholds_list = thresholds_list):
    device = logits.device
    B = logits.size(0)
    
    # 1) Your high-level OOD categories, in the same order as ood2id:
    categories = ["sadness", "joy", "fear", "anger"]
    
    # 2) Map each to its set of fine-grained GoEmotions IDs
    
    high2fine = {
        "anger":   ["anger", "annoyance", "disapproval"],
        "fear":    ["fear", "nervousness"],
        "joy":     ["joy", "amusement", "approval", "excitement",
                    "gratitude", "love", "optimism", "relief",
                    "pride", "admiration", "desire", "caring"],
        "sadness": ["sadness", "disappointment", "embarrassment",
                    "grief", "remorse"],
    }
    # build a list of id-lists in the same order as `categories`
    id_lists = [
        [emotion2id[emo] for emo in high2fine[cat]]
        for cat in categories
    ]

    
    #    (where id2emotion is your dict { id → emotion_name })
    thresholds_tensor = torch.tensor(thresholds_list, 
                                    device=logits.device, 
                                    dtype=logits.dtype)   # (28,)
    
    # 3) Fine-grained preds
    probs     = torch.sigmoid(logits)
    fine_pred = (probs >= thresholds_tensor).int()       # (B,28)
    
    # 4) Aggregate into 4-dim OOD preds
    pred_ood = torch.zeros((B, 4), dtype=torch.int, device=device)
    for idx, fine_ids in enumerate(id_lists):
        # if **any** child is predicted, flip that OOD dim on
        pred_ood[:, idx] = fine_pred[:, fine_ids].any(dim=1).int()
    
    # 5) Build true OOD labels
    true_ood = torch.zeros((B, 4), dtype=torch.int, device=device)
    for i, tl in enumerate(text_labels):
        # unwrap single-element lists
        if isinstance(tl, (list, tuple)) and len(tl) == 1:
            tl = tl[0]
        # if string, look up its index
        if isinstance(tl, str):
            idx = categories.index(tl)
        # if int, assume it’s already the correct OOD index
        elif isinstance(tl, int):
            idx = tl
        else:
            raise ValueError(f"Cannot interpret text_label={tl!r}")
        true_ood[i, idx] = 1
    
    return pred_ood, true_ood
    

def ood_evaluate(model, loader, device):
    model.eval()
    test_correct, n = 0, 0
    total_slots = 0
    exact_match = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            # labels = batch["labels"].to(device, non_blocking=True)
            text_labels=batch["text_labels"]
            logits = model(input_ids, attention_mask)

            preds, labels = ood_helper(logits, text_labels)
            # preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            total_slots += labels.numel()

            exact_match += (preds.eq(labels).all(dim=1)).float().sum().item()
            n += labels.size(0)

    # --- aggregate across processes ---
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor([test_correct, n, total_slots, exact_match], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        test_correct, n, total_slots, exact_match = tensor.tolist()

    return {"accuracy":test_correct / n, "hamming": test_correct / total_slots, 
            "subset": exact_match / n}


def evaluate(model, loader, device):
    model.eval()
    total_loss, test_correct, n = 0.0, 0, 0
    total_slots = 0
    exact_match = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
            # preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            total_slots += labels.numel()

            exact_match += (preds.eq(labels).all(dim=1)).float().sum().item()
            n += labels.size(0)

    # --- aggregate across processes ---
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor([total_loss, test_correct, n, total_slots, exact_match], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss, test_correct, n, total_slots, exact_match = tensor.tolist()

    return {"loss":total_loss / n, "accuracy":test_correct / n, 
            "hamming": test_correct / total_slots, "subset": exact_match / n}


def init_distributed_mode(local_rank: int):
    """Initialize torch.distributed if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size, rank = 1, 0
    return rank, world_size

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = parse_args()
    rank, world_size = init_distributed_mode(args.local_rank)
    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    
    set_seed(RNG_SEED + rank)

    if rank == 0:
        print(
            f"World size : {world_size}\nDevice: {device}\nBatch size: {args.batch_size}\nEpochs: {args.epochs}\n"
        )

    dfs = load_data()
    train_df, val_df, test_df = dfs['train'], dfs['val'],dfs['test']

    train_ds = GoEmotions(train_df, tokenizer, args.max_len)
    val_ds   = GoEmotions(val_df,   tokenizer, args.max_len)
    test_ds  = GoEmotions(test_df,  tokenizer, args.max_len)

    train_sampler = (
        DistributedSampler(train_df, shuffle=True) if world_size > 1 else None
    )
    val_sampler = DistributedSampler(val_df, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_df, shuffle=False) if world_size > 1 else None


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler = train_sampler,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
    )

    if rank == 0:
        print(f"Ensembling {args.ensemble} models")

    checkpoint = torch.load(args.model_loc, map_location = device)
    gpt2model = checkpoint['full_model']
    base_clf = GPT2EmoClassifier(model=gpt2model, num_classes=28, dropout=args.dropout)
# wrap it into an ensemble
    if args.ensemble > 1:
        model = EnsembleClassifier(base_clf, args.ensemble).to(device)
    else:
        model = base_clf.to(device)
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
        
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs #len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.06 * total_steps),
            num_training_steps=total_steps,
        )
    counts = train_df["labels"].explode().value_counts()  # count of each label index
    num_samples = len(train_df)
    pos_weight = torch.tensor([(num_samples - counts[i]) / counts[i] for i in range(28)]).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        train_correct, train_n = 0, 0
        correct, total_slots = 0, 0
        exact_match = 0
        train_loss = []
        metr_preds, metr_labs = [], []
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(input_ids, attention_mask)
            
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
            # preds = logits.argmax(dim=-1)
            metr_preds.append(list(preds.detach().cpu().numpy()))
            metr_labs.append(list(labels.detach().cpu().numpy()))
            batch_correct = (preds == labels).sum().item()
            batch_acc = batch_correct / labels.size(0)
            train_n += labels.size(0)
            train_correct += (preds == labels).sum().item()
            
            correct += (preds == labels).sum().item()
            total_slots += labels.numel()

            exact_match += (preds.eq(labels).all(dim=1)).float().sum().item()
            
            loss.backward()
            train_loss.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        if rank == 0:
            train_acc = train_correct / train_n
            hamming_acc = correct / total_slots
            subset_acc = exact_match / train_n
            print(f"{np.mean(train_loss)=:.4f} | {train_acc=:.4f} | {hamming_acc=:.4f} | {subset_acc=:.4f}")
            y_pred = np.vstack(metr_preds)   # shape (N, 28)
            y_true = np.vstack(metr_labs)    # shape (N, 28)
            print(f"f1 score: {f1_score(y_true, y_pred, average='macro'):.4f}")

        # ----- Baseline evaluation -----
    if args.no_ood:
        test_metr = evaluate(model, test_loader, device)
    else:
        test_metr = ood_evaluate(model, test_loader, device)
    if rank == 0:
        if args.no_ood:
            print(f"Epoch {epoch}  | TEST | loss {test_metr['loss']:.4f}  acc {test_metr['accuracy']:.3f} | hamming_acc {test_metr['hamming']:.3f} | subset_acc {test_metr['subset']:.3f}")
        else:
            print(f"Epoch {epoch}  | TEST | acc {test_metr['accuracy']:.3f} | hamming_acc {test_metr['hamming']:.3f} | subset_acc {test_metr['subset']:.3f}")
        
        repo_id = "sawlachintan/gpt2-goemotions-ft"
        repo_dir = "gpt2-goemotions-ft"
        repo = Repository(local_dir=repo_dir, clone_from=f"https://huggingface.co/{repo_id}", use_auth_token=True)
        output_type = ""
        output_type += "no-ood-" if args.no_ood else "ood-"
        output_type += f"epochs-{args.epochs}-"
        output_type += f"ense-{args.ensemble}"
        repo_dir += output_type
        os.makedirs(repo_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model

        torch.save(model_to_save.state_dict(), os.path.join(repo_dir, "pytorch_model.bin"))

        tokenizer.save_pretrained(repo_dir)

        with open(os.path.join(repo_dir, "README.md"), "w") as f:
            f.write("# Fine-tuned GPT2EmoClassifier on GoEmotions\nThis repo contains a PyTorch state dict and tokenizer.")

    cleanup_distributed()

if __name__ == "__main__":
    main()