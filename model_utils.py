import os
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from HF_FineTuned_GPT2 import MyGPT2
import torch

def load_tokenizer_and_model(model_cache: str, model_name: str):
    """Download (or load from `model_cache`) a GPT2ForSequenceClassification + its tokenizer."""
    if os.path.isdir(model_cache):
        tok = GPT2Tokenizer.from_pretrained(model_cache, local_files_only=True)
        mdl = GPT2ForSequenceClassification.from_pretrained(model_cache, local_files_only=True)
    else:
        tok = GPT2Tokenizer.from_pretrained(model_name)
        mdl = GPT2ForSequenceClassification.from_pretrained(model_name)
        tok.save_pretrained(model_cache)
        mdl.save_pretrained(model_cache)
    tok.pad_token_id = tok.eos_token_id
    return tok, mdl

def build_wrapped_model(tokenizer, pretrained_model, device: str):
    """Wrap in your MyGPT2 and move to device."""
    model = MyGPT2(tokenizer=tokenizer, model=pretrained_model)
    model.to(device)
    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.hf_model.parameters())
    trainable = sum(p.numel() for p in model.hf_model.parameters() if p.requires_grad)
    return total, trainable

def freeze_backbone(model):
    """
    Freeze GPT2 backbone; only leave classification head trainable.
    """
    for param in model.hf_model.transformer.parameters():
        param.requires_grad = False
    for param in model.hf_model.score.parameters():
        param.requires_grad = True


