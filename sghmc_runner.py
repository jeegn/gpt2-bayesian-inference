import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

def sghmc_sampler(model, train_loader, device, num_samples=100, burn_in=50, lr=1e-4, noise_std=1e-4):
    """
    Perform SGHMC on the model's classification head only.

    Args:
        model: your GPT2 model with classification head
        train_loader: dataloader (we'll loop over it)
        device: cuda or cpu
        num_samples: number of posterior samples to collect
        burn_in: how many steps to ignore for burn-in
        lr: learning rate for SGHMC
        noise_std: standard deviation of injected noise

    Returns:
        sampled_models: list of model snapshots (classifier head state_dicts)
    """
    model.train()
    classifier = model.hf_model.score  # Assuming score is the final layer

    optimizer = SGHMC(classifier.parameters(), lr=lr, noise=noise_std)
    sampled_models = []
    step = 0

    while len(sampled_models) < num_samples:
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1
            if step > burn_in:
                sampled_models.append(deepcopy(classifier.state_dict()))
                if len(sampled_models) >= num_samples:
                    break

    return sampled_models

class SGHMC(torch.optim.Optimizer):
    """
    Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) with momentum buffers.
    """
    def __init__(self, params, lr=1e-4, noise=1e-4, momentum_decay=0.01):
        """
        Args:
            params: parameters to optimize
            lr: learning rate (step size)
            noise: standard deviation of injected noise
            momentum_decay: controls friction / damping
        """
        defaults = dict(lr=lr, noise=noise, momentum_decay=momentum_decay)
        super(SGHMC, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            noise_std = group['noise']
            momentum_decay = group['momentum_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # State initialization
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # SGHMC update
                noise = torch.randn_like(p.grad) * noise_std
                buf.mul_(1 - momentum_decay).add_(-lr * (p.grad + noise))
                p.add_(buf)

def predict_sghmc(model, sampled_models, dataloader, device):
    """
    Predict by averaging over SGHMC samples.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for state in sampled_models:
            model.hf_model.score.load_state_dict(state)
            batch_logits = []
            for batch in dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(inputs)
                batch_logits.append(logits.cpu().numpy())
            all_preds.append(np.vstack(batch_logits))

    mean_logits = np.mean(np.stack(all_preds, axis=0), axis=0)
    return mean_logits
