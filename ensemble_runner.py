import os
import torch
import numpy as np
from gpt2_emo_config import GPT, GPTConfig, GPT2EmoClassifier

def load_single_ensemble_model(model_path, device):
    """
    Load a single GPT2EmoClassifier model from a saved .bin checkpoint.
    """
    config = GPTConfig()
    gpt_model = GPT(config)
    model = GPT2EmoClassifier(model=gpt_model)
    
    # Load the checkpoint
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def load_ensemble_models(base_folder, n_models, device):
    """
    Load N models from checkpoints saved in base_folder.
    
    Expects filenames like:
      base_folder/ensemble_0/pytorch_model.bin
      base_folder/ensemble_1/pytorch_model.bin
      ...
    """
    models = []
    for i in range(n_models):
        model_path = os.path.join(base_folder, f"gpt2-goemotions-ft-ood-epochs-10-ense-1inc-{i}", "pytorch_model.bin")
        model = load_single_ensemble_model(model_path, device)
        models.append(model)
    return models

def ensemble_inference(models, dataloader, device):
    """
    Predict by averaging logits across ensemble models.
    """
    all_logits = []

    for model in models:
        model_preds = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(inputs["input_ids"], attention_mask=inputs.get("attention_mask", None))
                model_preds.append(logits.cpu().numpy())
        model_preds = np.vstack(model_preds)
        all_logits.append(model_preds)

    # Average logits
    mean_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    return mean_logits