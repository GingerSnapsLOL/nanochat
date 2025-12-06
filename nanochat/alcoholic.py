"""
Alcoholic version of nanochat Model
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from functools import partial
from nanochat.gpt import apply_rotary_emb
from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW, AdamW


@dataclass
class AlcoholicNanoConfig:
    sequence_len: int = 2048
    vocab_size: int = 50000

    # core size
    n_layer: int = 30          # careful: huge for 16GB, see below
    n_embd: int = 1536
    n_head: int = 12           # head_dim = 1536 / 12 = 128
    n_kv_head: int = 4         # GQA ratio = 3
    head_dim: int = None       # computed from n_embd // n_head if None (for backward compatibility)

    # rotary / attention tricks
    rope_theta: float = 1_000_000.0
    qk_norm: bool = True

    # norms & MLP
    norm_type: str = "rmsnorm"
    mlp_type: str = "swiglu"
    ffn_mult: float = 3.5      # use this instead of fixed intermediate_size
    intermediate_size: int = None  # computed from ffn_mult if None

    # regularization & scaling
    attn_dropout: float = 0.0
    resid_dropout: float = 0.1
    mlp_dropout: float = 0.0
    resid_scale: float = 1.0
    
    def __post_init__(self):
        """Compute head_dim if not provided (for backward compatibility with old checkpoints)."""
        if self.head_dim is None:
            self.head_dim = self.n_embd // self.n_head 

class AlcoholicRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight




class AlcoholicMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        if config.intermediate_size is None:
            h = int(config.ffn_mult * d)
        else:
            h = config.intermediate_size
        if config.mlp_type == "swiglu":
            self.w12 = nn.Linear(d, 2 * h, bias=False)
            self.out = nn.Linear(h, d, bias=False)
            self.use_swiglu = True
            self.mlp_dropout = nn.Dropout(config.mlp_dropout)
        else:
            self.up = nn.Linear(d, h, bias=False)
            self.act = nn.GELU()
            self.down = nn.Linear(h, d, bias=False)
            self.use_swiglu = False
            self.mlp_dropout = nn.Dropout(config.mlp_dropout)
        
        # For backward compatibility with old checkpoints that use self.dropout
        self.dropout = self.mlp_dropout

    def forward(self, x):
        if self.use_swiglu:
            a, b = self.w12(x).chunk(2, dim=-1)
            return self.dropout(self.out(F.silu(a) * b))
        else:
            return self.dropout(self.down(self.act(self.up(x))))


class AlcoholicCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        # head_dim is computed in config.__post_init__ if not provided (for backward compatibility)
        self.head_dim = config.head_dim
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.use_rope = True
        self.qk_norm = config.qk_norm

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.attn_dropout = nn.Dropout(config.attn_dropout)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # project to Q/K/V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # apply RoPE
        cos, sin = cos_sin
        if self.use_rope:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # optional QK norm
        if self.qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

        # [B,T,H,D] -> [B,H,T,D] (and HK for k,v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # KV cache integration
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)
        Tk = k.size(2)
        enable_gqa = self.n_head != self.n_kv_head

        # attention modes: training / single-token decode / chunked decode
        if kv_cache is None or Tq == Tk:
            # training or full-prefix chunk
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                enable_gqa=enable_gqa,
            )
        elif Tq == 1:
            # single-token decode attends to full prefix
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
        else:
            # chunked decode: prefix fully visible, causal inside chunk
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)  # False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                enable_gqa=enable_gqa,
            )

        # back to [B,T,C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y





class AlcoholicBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = AlcoholicCausalSelfAttention(config, layer_idx)
        self.mlp = AlcoholicMLP(config)
        self.norm1 = AlcoholicRMSNorm(config.n_embd)
        self.norm2 = AlcoholicRMSNorm(config.n_embd)
        self.resid_scale = config.resid_scale or (1.0 / (2 * config.n_layer) ** 0.5)
        
    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(self.norm1(x), cos_sin, kv_cache) * self.resid_scale
        x = x + self.mlp(self.norm2(x)) * self.resid_scale
        return x



class AlcoholicNanoGPT(nn.Module):
    def __init__(self, config: AlcoholicNanoConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([AlcoholicBlock(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Rotary cache (same pattern as nanochat, but with your theta)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim, base=config.rope_theta
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.norm_in = AlcoholicRMSNorm(config.n_embd)
        self.norm_out = AlcoholicRMSNorm(config.n_embd)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction: str = "mean"):
        B, T = idx.size()

        # Rotary embeddings sanity checks (same pattern as nanochat GPT)
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"idx on {idx.device}, rotary on {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be bfloat16"

        # If we have a KV cache (inference), offset RoPE by current cache position
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]

        # Embedding + input norm
        x = self.transformer.wte(idx)           # (B, T, C)
        x = self.norm_in(x)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)

        # Final norm
        x = self.norm_out(x)

        # LM head + logits softcap (like nanochat GPT)
        softcap = 15.0
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)

        # Training mode: return loss
        if targets is not None:
            logits = logits.float()  # ensure fp32 / tf32 for CE
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss

        # Inference mode: return logits
        return logits


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / (fan_in ** 0.5) * min(1.0, (fan_out / fan_in) ** 0.5)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
        elif isinstance(module, AlcoholicRMSNorm):
            nn.init.ones_(module.weight)


    def init_weights(self):
        self.apply(self._init_weights)
        # zero-out lm_head and proj weights like nanochat GPT
        nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            # MLP out / down
            if getattr(block.mlp, "use_swiglu", False):
                nn.init.zeros_(block.mlp.out.weight)
            else:
                nn.init.zeros_(block.mlp.down.weight)
            nn.init.zeros_(block.attn.c_proj.weight)

        # recompute rotary embeddings on correct device
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim, base=self.config.rope_theta
        )
        self.cos, self.sin = cos, sin

        # cast embeddings to bf16 on cuda (like GPT)
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)


    def estimate_flops(self):
        """Estimated FLOPs per token (same formula as nanochat GPT)."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l = self.config.n_layer
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token



    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2,
                         matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # split params
        # Alcoholic model has learnable RMSNorm parameters (1D tensors) that must go to AdamW
        # Muon only handles 2D+ tensors (matrices), so we need to filter 1D params
        all_transformer_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        # norm_in and norm_out are 1D (weight vectors)
        norm_in_out_params = list(self.norm_in.parameters()) + list(self.norm_out.parameters())
        
        # Filter transformer params: separate 2D (matrices) from 1D (norm weights)
        # AlcoholicBlock has norm1 and norm2 which are 1D, these must go to AdamW
        matrix_params = []  # 2D+ tensors for Muon
        norm_block_params = []  # 1D norm weights from blocks, for AdamW
        
        for p in all_transformer_params:
            if p.ndim >= 2:
                matrix_params.append(p)  # Linear layer weights (2D)
            else:
                norm_block_params.append(p)  # Norm weights (1D)
        
        # All norm parameters (from blocks + top-level)
        all_norm_params = norm_block_params + norm_in_out_params
        
        # Verify all parameters are accounted for
        all_params = list(self.parameters())
        accounted_params = matrix_params + embedding_params + lm_head_params + all_norm_params
        if len(all_params) != len(accounted_params):
            # Debug: show what's missing
            all_param_names = {id(p): name for name, p in self.named_parameters()}
            accounted_param_ids = {id(p) for p in accounted_params}
            missing = [all_param_names[id(p)] for p in all_params if id(p) not in accounted_param_ids]
            raise AssertionError(
                f"Parameter count mismatch: total={len(all_params)}, accounted={len(accounted_params)}\n"
                f"Missing parameters: {missing}"
            )

        # LR scale by d_model, same as GPT
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
            print0(f"Muon params (2D+): {len(matrix_params)}, AdamW params (1D): {len(all_norm_params) + len(embedding_params) + len(lm_head_params)}")

        # AdamW groups: lm_head, embeddings, and ALL norm layers (all 1D or need AdamW)
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=all_norm_params, lr=embedding_lr * dmodel_lr_scale),  # All norm layers use embedding LR
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers


    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming generation, batch size = 1.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)        # (B, T, vocab)
            logits = logits[:, -1, :]         # last step

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()