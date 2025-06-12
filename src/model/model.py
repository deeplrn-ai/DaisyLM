import torch
from torch import nn

class ModelConfig:
    vocab_size = 32000
    d_model = 768
    n_layers = 12
    n_heads = 12
    d_head = 64
    intermediate_size = 3072
    d_latent = 32
    d_rope_sub = 16

class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))

    def __call__(self, x):
        norm = (x.pow(2).mean(-1, keepdims=True) + self.eps).sqrt()
        out = x / norm
        return out * self.weight


def precompute_freq_cis(dim: int, seq_len: int, theta: float = 10000.0):
    base_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=base_freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, base_freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    q_c = torch.view_as_complex(q.float().reshape(q.shape[:-1] + (q.shape[-1] // 2, 2)))
    k_c = torch.view_as_complex(k.float().reshape(k.shape[:-1] + (k.shape[-1] // 2, 2)))
    seq_len = q.shape[2]
    freq_cis = freqs_cis[None, None, :seq_len, :]

    q_r = torch.view_as_real(q_c * freq_cis).flatten(3)
    k_r = torch.view_as_real(k_c * freq_cis).flatten(3)

    return q_r.type_as(q), k_r.type_as(k)



