# """Your algorithm goes here. The default is a complete, runnable baseline.
#
# Required work: diagnose a limitation and implement a structural/training/memory
# change. Explain it, measure its cost and perform a mechanism ablation. Merely
# renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
# You can replace this factory/model completely while keeping the two model interfaces.
# """


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]
        return (
            self.cos_cached[:seq_len, :].to(x.device),
            self.sin_cached[:seq_len, :].to(x.device),
        )

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class SwiGLU(nn.Module):
    def __init__(self, width, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * width) / 3)
            hidden_dim = ((hidden_dim + 31) // 32) * 32
        self.w1 = nn.Linear(width, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, width, bias=False)
        self.w3 = nn.Linear(width, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self, width=256, heads=8, dropout=0.0, rope=True, swiglu=True):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.rope = rope
        self.swiglu = swiglu

        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)

        if swiglu:
            self.mlp = SwiGLU(width, dropout=dropout)
        else:
            # 消融对照组：标准 GELU MLP
            self.mlp = nn.Sequential(
                nn.Linear(width, 4 * width),
                nn.GELU(),
                nn.Linear(4 * width, width),
                nn.Dropout(dropout)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope=None):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope and rope is not None:
            cos, sin = rope(x)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        drop_p = self.dropout.p if self.training else 0.0
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        attended = attended.transpose(1, 2).reshape(batch, length, width)

        x = x + self.dropout(self.proj(attended))
        x = x + self.mlp(self.norm2(x))
        return x

class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config.get('context', 256)
        width = config.get('width', 256)
        heads = config.get('heads', 8)
        depth = config.get('depth', 8)
        dropout = config.get('dropout', 0.05)
        vocab = config.get('vocab', 2048)

        # 消融开关控制标志
        self.rope = config.get('rope', True)
        self.swiglu = config.get('swiglu', True)
        self.weight_tying = config.get('weight_tying', True)

        self.token = nn.Embedding(vocab, width)
        if self.rope:
            self.rope = RotaryEmbedding(width // heads, max_seq_len=self.context)
        else:
            # 消融对照组：常规绝对位置编码
            self.pos_emb = nn.Embedding(self.context, width)
            self.rope = None

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(width, heads, dropout, rope=self.rope, swiglu=self.swiglu)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

        self.apply(self.initialize)

        if self.weight_tying:
            self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def features(self, ids):
        b, t = ids.shape
        x = self.token(ids)
        if not self.rope:
            pos = torch.arange(0, t, device=ids.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, self.rope)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)

def build_model(config):
    cfg = dict(config)
    cfg['width'] = 256
    cfg['depth'] = 8
    cfg['heads'] = 8

    # 控制消融实验
    cfg.setdefault('rope', True)
    cfg.setdefault('swiglu', True)
    cfg.setdefault('weight_tying', False)

    return StudentGPT(cfg)