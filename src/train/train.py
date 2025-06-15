import torch
from torch.cuda import amp
from torch import optim
import tiktoken
from transformers import get_cosine_schedule_with_warmup, opt
from src.model.model import Model, ModelConfig

config = ModelConfig()
model = Model(config)

