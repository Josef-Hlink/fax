# fax/models/gpt_v5_1.py
# Minimal blocks needed for GPTv5_1Controller (relative-pos attention + MLP heads).
# Expects:
#   - preprocessor.input_size: int
#   - preprocessor.data_config.{num_stages,stage_embedding_dim,
#                               num_characters,character_embedding_dim,
#                               num_actions,action_embedding_dim}
#   - preprocessor.target_config.target_shapes_by_head: dict with keys
#       {"c_stick","main_stick","buttons","shoulder"} -> (out_dim, ...)
#
# Inputs (TensorDict) are expected to contain:
#   "stage", "ego_character", "opponent_character",
#   "ego_action", "opponent_action", "gamestate", "controller"

from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from fax.constants import (
    NUM_STAGES,
    STAGE_EMBEDDING_DIM,
    NUM_CHARACTERS,
    CHARACTER_EMBEDDING_DIM,
    NUM_ACTIONS,
    ACTION_EMBEDDING_DIM,
)
from fax.processing.preprocessor import Preprocessor
from fax.config import DataConfig


@dataclass
class GPTConfig:
    block_size: int
    n_embd: int
    n_layer: int
    n_head: int
    dropout: float = 0.0
    bias: bool = True  # bias in Linear/LayerNorm (GPT-2 style)


class Model(nn.Module):
    """
    GPT with 4 MLP heads for controller components.
    Order: c_stick -> main_stick -> buttons -> shoulder (each head sees previous via detached concat).
    """

    def __init__(self, preprocessor: Preprocessor, gpt_config: GPTConfig) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.gpt_config = gpt_config
        self.block_size = gpt_config.block_size
        self.input_size = preprocessor.input_size
        self.n_embd = gpt_config.n_embd

        self.stage_emb = nn.Embedding(NUM_STAGES, STAGE_EMBEDDING_DIM)
        self.character_emb = nn.Embedding(NUM_CHARACTERS, CHARACTER_EMBEDDING_DIM)
        self.action_emb = nn.Embedding(NUM_ACTIONS, ACTION_EMBEDDING_DIM)

        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, gpt_config.n_embd),
                drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList(
                    [_BlockRelativePosition(gpt_config) for _ in range(gpt_config.n_layer)]
                ),
                ln_f=nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

        tgt = preprocessor.target_config.target_shapes_by_head
        c_stick_in = self.n_embd
        c_stick_out = tgt['c_stick'][0]

        main_in = self.n_embd + c_stick_out
        main_out = tgt['main_stick'][0]

        btn_in = self.n_embd + c_stick_out + main_out
        btn_out = tgt['buttons'][0]

        shoulder_in = self.n_embd + c_stick_out + main_out + btn_out
        shoulder_out = tgt['shoulder'][0]

        self.c_stick_head = nn.Sequential(
            nn.LayerNorm(c_stick_in, bias=gpt_config.bias),
            nn.Linear(c_stick_in, c_stick_in // 2),
            nn.GELU(),
            nn.Linear(c_stick_in // 2, c_stick_out),
        )
        self.main_stick_head = nn.Sequential(
            nn.LayerNorm(main_in, bias=gpt_config.bias),
            nn.Linear(main_in, main_in // 2),
            nn.GELU(),
            nn.Linear(main_in // 2, main_out),
        )
        self.button_head = nn.Sequential(
            nn.LayerNorm(btn_in, bias=gpt_config.bias),
            nn.Linear(btn_in, btn_in // 2),
            nn.GELU(),
            nn.Linear(btn_in // 2, btn_out),
        )
        self.shoulder_head = nn.Sequential(
            nn.LayerNorm(shoulder_in, bias=gpt_config.bias),
            nn.Linear(shoulder_in, shoulder_in // 2),
            nn.GELU(),
            nn.Linear(shoulder_in // 2, shoulder_out),
        )

        # heads are new modules; init their weights
        for m in [self.c_stick_head, self.main_stick_head, self.button_head, self.shoulder_head]:
            for mod in m.modules():
                self._init_weights(mod)

    def forward(self, inputs: TensorDict) -> TensorDict:
        B, L, _ = inputs['gamestate'].shape
        assert L <= self.block_size, f'seq len {L} > block_size {self.block_size}'

        x = self._embed_inputs(inputs)  # (B, L, G_cat)
        x = self.transformer.proj_down(x)  # (B, L, D)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (B, L, D)

        # decode with detached chaining
        c_stick = self.c_stick_head(x)
        main_stick = self.main_stick_head(torch.cat([x, c_stick.detach()], dim=-1))
        buttons = self.button_head(torch.cat([x, c_stick.detach(), main_stick.detach()], dim=-1))
        shoulder = self.shoulder_head(
            torch.cat([x, c_stick.detach(), main_stick.detach(), buttons.detach()], dim=-1)
        )

        return TensorDict(
            {
                'buttons': buttons,
                'main_stick': main_stick,
                'c_stick': c_stick,
                'shoulder': shoulder,
            },
            batch_size=(B, L),
        )

    # private helper methods
    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        """Expects TensorDict with following keys:
        'stage', 'ego_character', 'opponent_character',
        'ego_action', 'opponent_action', 'gamestate', 'controller'
        """
        return torch.cat(
            [
                self.stage_emb(inputs['stage']).squeeze(-2),
                self.character_emb(inputs['ego_character']).squeeze(-2),
                self.character_emb(inputs['opponent_character']).squeeze(-2),
                self.action_emb(inputs['ego_action']).squeeze(-2),
                self.action_emb(inputs['opponent_action']).squeeze(-2),
                inputs['gamestate'],
                inputs['controller'],
            ],
            dim=-1,
        )


class _MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def skew(QEr: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient "skewing" trick to avoid materializing O(T^2) `R` matrix.

    Music Transformer, Huang et al. (2018) https://arxiv.org/abs/1809.04281
    Implementation by: https://jaketae.github.io/study/relative-positional-encoding/
    """
    # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
    padded = F.pad(QEr, (1, 0))
    # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
    batch_size, num_heads, num_rows, num_cols = padded.shape
    reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
    # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
    Srel = reshaped[:, :, 1:, :]
    # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
    return Srel


class _CausalSelfAttentionRelativePosition(nn.Module):
    """Self-attention with Shaw-style relative position encodings, causal mask."""

    def __init__(self, config: GPTConfig, input_size: int | None = None) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.block_size = config.block_size
        self.n_embd = input_size or config.n_embd
        self.n_head = config.n_head
        self.hs = self.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.Er = nn.Parameter(torch.randn(self.block_size, self.hs))  # (L, hs)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.register_buffer(
            'bias',
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        assert L <= self.block_size, f'seq len {L} > block_size {self.block_size}'

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, L, self.n_head, self.hs).transpose(1, 2)  # (B, H, L, hs)
        k = k.view(B, L, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.hs).transpose(1, 2)

        # relative position
        start = self.block_size - L
        Er_t = self.Er[start:, :].transpose(0, 1)  # (hs, L)
        QEr = q @ Er_t  # (B, H, L, L)
        Srel = skew(QEr)  # (B, H, L, L)

        scale = 1.0 / math.sqrt(self.hs)
        att = (q @ k.transpose(-2, -1) + Srel) * scale  # (B, H, L, L)
        att = att.masked_fill(self.bias[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, H, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.resid_dropout(self.c_proj(y))
        return y


class _BlockRelativePosition(nn.Module):
    """A single Transformer block with relative position attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = _CausalSelfAttentionRelativePosition(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = _MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == '__main__':
    # simple test with random tensors

    model = Model(
        preprocessor=Preprocessor(DataConfig('~/Data/mds/full')),
        gpt_config=GPTConfig(
            block_size=16,
            n_embd=128,
            n_layer=2,
            n_head=4,
            dropout=0.1,
            bias=True,
        ),
    )
    print(model.preprocessor.stats)
    B, L = 2, 10
    inputs = TensorDict(
        {
            'stage': torch.randint(0, NUM_STAGES, (B, 1)),
            'ego_character': torch.randint(0, NUM_CHARACTERS, (B, 1)),
            'opponent_character': torch.randint(0, NUM_CHARACTERS, (B, 1)),
            'ego_action': torch.randint(0, NUM_ACTIONS, (B, 1)),
            'opponent_action': torch.randint(0, NUM_ACTIONS, (B, 1)),
            'gamestate': torch.randn(
                B, L, model.preprocessor.input_size - 2 * CHARACTER_EMBEDDING_DIM
            ),
            'controller': torch.randn(
                B, L, model.preprocessor.input_size - 2 * CHARACTER_EMBEDDING_DIM
            ),
        },
        batch_size=(B,),
    )
    outputs = model(inputs)
    for k, v in outputs.items():
        print(k, v.shape)
    # should be (B, L, out_dim) for each head
    print('Model test passed.')
