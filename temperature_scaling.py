import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ModelWithTemperature(nn.Module):
    """
    A thin wrapper around a model that applies temperature scaling to logits.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, inputs):
        logits = self.model(inputs)
        return logits / self.temperature

    def set_temperature(self, valid_loader, device):
        """
        Tune the temperature of the model using the validation set.
        """
        self.to(device)
        self.model.eval()
        nll_criterion = nn.BCEWithLogitsLoss()

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in valid_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].float().to(device)

                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        print(f"Optimal temperature: {self.temperature.item():.4f}")

        return self

