"""Pre-store only the weights that CUDA autocast already casts to BF16.

Embedding and normalization parameters stay FP32. A blanket model.bfloat16()
does not preserve those semantics and is deliberately not used. Checkpoints
remain untouched and FP32; this is an inference-only in-memory transformation.
"""
import torch
from torch import nn


def precast_autocast_weights(model):
    if model.training: raise ValueError('Precasting is inference-only; call eval first')
    for module in model.modules():
        if isinstance(module,(nn.Conv1d,nn.Conv3d,nn.Linear)):
            module.to(dtype=torch.bfloat16)
    return model
