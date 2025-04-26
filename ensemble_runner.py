import torch
import numpy as np

def ensemble_inference(models, dataloader, device):
    all_logits = []
    for model in models:
        model.eval()
        batch_logits = []
        with torch.no_grad():
            for batch in dataloader:
                data = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(data)
                batch_logits.append(logits.cpu().numpy())
            all_logits.append(np.vstack(batch_logits))
    # Average logits across models
    mean_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    return mean_logits