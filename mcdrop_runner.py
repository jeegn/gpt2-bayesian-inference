import torch
import numpy as np

def mc_dropout_inference(model, dataloader, device, n_forward_passes=30):
    model.train()  # Enable dropout!
    all_logits = []
    with torch.no_grad():
        for _ in range(n_forward_passes):
            batch_logits = []
            for batch in dataloader:
                data = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(data)  # Forward pass
                batch_logits.append(logits.cpu().numpy())
            all_logits.append(np.vstack(batch_logits))
    # Average logits across passes
    mean_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    return mean_logits

def check_dropout_layers(model):
    has_dropout = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            has_dropout = True
            print(f"Dropout layer found: {module}")
    if not has_dropout:
        print("No Dropout layers found in model!")
    else:
        print("Dropout layers exist.")