import torch
from torch import nn
import torch.nn.functional as F

class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64
    intermediate_size: int = 3072
    d_latent: int = 32
    d_rope_sub: int = 16
    attn_dropout: float = 0.1

class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))

    def __call__(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freq_cis(dim: int, seq_len: int, theta: float = 10000.0):
    base_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=base_freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, base_freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_c = torch.view_as_complex(x.float().reshape(x.shape[:-1] + (x.shape[-1] // 2, 2)))
    seq_len = x.shape[2]
    freq_cis = freqs_cis[None, None, :seq_len, :]

    x_r = torch.view_as_real(x_c * freq_cis).flatten(3)

    return x_r.type_as(x)


class MLA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_latent = config.d_latent
        self.d_rope_sub = config.d_rope_sub
        self.d_nope_sub = self.d_head - self.d_rope_sub
        self.attn_dropout = nn.Dropout(config.attn_dropout)

        self.kv_latent_norm = RMSNorm(self.d_model)

        self.wq = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=True)
        self.wkv_a = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.wkv_b = nn.Linear(self.d_latent, 2 * self.n_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=True)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, past_key_values=None, use_cache=None):
        batch_size, seq_len, _ = x.shape
        current_latent_kv = self.kv_latent_norm(self.wkv_a(x))

        if past_key_values is None:
            cache_latent_kv = current_latent_kv
            past_seq_len = 0
        else:
            cache_latent_kv = torch.cat([past_key_values, current_latent_kv], dim=1)
            past_seq_len = past_key_values.shape[1]

        q_position_ids = torch.arange(past_seq_len, past_seq_len + seq_len, dtype=torch.long, device=x.device)
        k_position_ids = torch.arange(0, past_seq_len + seq_len, dtype=torch.long, device=x.device)
        updated_freqs_cis_q = freqs_cis[q_position_ids]
        updated_freqs_cis_k = freqs_cis[k_position_ids]

        updated_kv_for_layer = cache_latent_kv
        kv = self.wkv_b(updated_kv_for_layer)

        q = self.wq(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        q_nope, q_rope = torch.split(q, [self.d_nope_sub, self.d_rope_sub], dim=-1)
        q_roped = apply_rotary_emb(q_rope, updated_freqs_cis_q)
        q = torch.cat([q_nope, q_roped], dim=-1)

        k, v = torch.chunk(kv, chunks=2, dim=-1)
        k = k.view(batch_size, -1, self.n_heads, self.d_head)
        k_nope, k_rope = torch.split(k, [self.d_nope_sub, self.d_rope_sub], dim=-1)
        k_roped = apply_rotary_emb(k_rope, updated_freqs_cis_k)
        k = torch.cat([k_nope, k_roped], dim=-1)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        try:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0)
        except Exception as e:
            print(f"Warning: F.scaled_dot_product_attention failed ({e}). Falling back to manual attention.")
            T_q = q.shape[-2]
            T_k = k.shape[-2]
        
            scale = 1 / (q.shape[-1] ** 0.5)
            attn_weight = q @ k.transpose(-2, -1) * scale
            
            mask_ones = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril(diagonal=0)

            if T_q == T_k:
                attn_weight = attn_weight.masked_fill(mask_ones.logical_not(), float('-inf'))
            else:
                attn_mask = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device)
                for i in range(T_q):
                    attn_mask[i, i + past_seq_len + 1:] = 0
                attn_weight = attn_weight.masked_fill(attn_mask.logical_not(), float('-inf')) 
            
            attn_weight = F.softmax(attn_weight, dim=-1)
            attn_weight = self.attn_dropout(attn_weight)
            attn = attn_weight @ v

        attn_output = attn.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_head)
        
        final_output = self.out_proj(attn_output)
        final_output = self.attn_dropout(final_output)

        if use_cache:
            return final_output, updated_kv_for_layer
        return final_output 


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, dim)
        self.l3 = nn.Linear(dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        return self.l2(F.silu(self.l1(x)) * self.l3(x))
