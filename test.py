import torch
import torch.nn as nn
from gpt2_emo_config import GPT, GPTConfig, GPT2EmoClassifier

# First, explicitly create a matching config and GPT model:
config = GPTConfig()
gpt_model = GPT(config)

# Now, wrap it in the classifier
model = GPT2EmoClassifier(model=gpt_model)
state_dict = torch.load("/scratch/scholar/jdani/project/ensemble_models/gpt2-goemotions-ft-ood-epochs-10-ense-1inc-0/pytorch_model.bin", map_location='cuda')

model.load_state_dict(state_dict)

print(model)