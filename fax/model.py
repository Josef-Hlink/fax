from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from fax.config import create_parser, parse_args, CFG
from fax.constants import (
    ACTION_EMBEDDING_DIM,
    CHARACTER_EMBEDDING_DIM,
    NUM_ACTIONS,
    NUM_CHARACTERS,
    NUM_STAGES,
    STAGE_EMBEDDING_DIM,
)
from fax.dataprep.stats import load_dataset_stats
from fax.processing.preprocessor import Preprocessor


class Model(nn.Module):
    """
    GPT with 4 MLP heads for controller components.
    Order: c_stick -> main_stick -> buttons -> shoulder (each head sees previous via detached concat).
    """

    def __init__(self, preprocessor: Preprocessor, cfg: CFG) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.block_size = cfg.model.seq_len
        self.input_size = preprocessor.input_size
        self.n_embd = cfg.model.emb_dim

        self.stage_emb = nn.Embedding(NUM_STAGES, STAGE_EMBEDDING_DIM)
        self.character_emb = nn.Embedding(NUM_CHARACTERS, CHARACTER_EMBEDDING_DIM)
        self.action_emb = nn.Embedding(NUM_ACTIONS, ACTION_EMBEDDING_DIM)

        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, cfg.model.emb_dim),
                drop=nn.Dropout(cfg.model.dropout),
                h=nn.ModuleList([_BlockRelativePosition(cfg) for _ in range(cfg.model.n_layers)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=True),
            )
        )

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.model.n_layers))

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
            nn.LayerNorm(c_stick_in, bias=True),
            nn.Linear(c_stick_in, c_stick_in // 2),
            nn.GELU(),
            nn.Linear(c_stick_in // 2, c_stick_out),
        )
        self.main_stick_head = nn.Sequential(
            nn.LayerNorm(main_in, bias=True),
            nn.Linear(main_in, main_in // 2),
            nn.GELU(),
            nn.Linear(main_in // 2, main_out),
        )
        self.button_head = nn.Sequential(
            nn.LayerNorm(btn_in, bias=True),
            nn.Linear(btn_in, btn_in // 2),
            nn.GELU(),
            nn.Linear(btn_in // 2, btn_out),
        )
        self.shoulder_head = nn.Sequential(
            nn.LayerNorm(shoulder_in, bias=True),
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
        x = cast(nn.Linear, self.transformer['proj_down'])(x)  # (B, L, D)
        x = cast(nn.Dropout, self.transformer['drop'])(x)
        for block in cast(nn.ModuleList, self.transformer['h']):
            x = block(x)
        x = cast(nn.LayerNorm, self.transformer['ln_f'])(x)  # (B, L, D)

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
    def __init__(self, cfg: CFG) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.model.emb_dim, 4 * cfg.model.emb_dim, bias=True)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.model.emb_dim, cfg.model.emb_dim, bias=True)
        self.dropout = nn.Dropout(cfg.model.dropout)

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

    def __init__(self, cfg: CFG) -> None:
        super().__init__()
        assert cfg.model.emb_dim % cfg.model.n_heads == 0
        self.cfg = cfg
        self.block_size = cfg.model.seq_len
        self.n_embd = cfg.model.emb_dim
        self.n_heads = cfg.model.n_heads
        self.hs = self.n_embd // cfg.model.n_heads
        self.dropout = cfg.model.dropout

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.Er = nn.Parameter(torch.randn(self.block_size, self.hs))  # (L, hs)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.bias: torch.Tensor
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
        q = q.view(B, L, self.n_heads, self.hs).transpose(1, 2)  # (B, H, L, hs)
        k = k.view(B, L, self.n_heads, self.hs).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.hs).transpose(1, 2)

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

    def __init__(self, cfg: CFG) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.model.emb_dim, bias=True)
        self.attn = _CausalSelfAttentionRelativePosition(cfg)
        self.ln_2 = nn.LayerNorm(cfg.model.emb_dim, bias=True)
        self.mlp = _MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == '__main__':
    # simple test with random tensors
    exposed_args = {
        'BASE': 'debug',
        'MODEL': 'n_layers n_heads seq_len emb_dim dropout gamma',
        'TRAINING': 'batch_size',
    }
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)
    model = Model(Preprocessor(cfg, load_dataset_stats(cfg.paths.mds / 'onefox')), cfg)

    B, L = cfg.training.batch_size, cfg.model.seq_len
    # Get the correct dimensions from the input config
    gamestate_dim = model.preprocessor.input_config.input_shapes_by_head['gamestate'][0]  # 18
    controller_dim = model.preprocessor.input_config.input_shapes_by_head['controller'][0]  # 48

    inputs = TensorDict(
        {
            'stage': torch.randint(0, NUM_STAGES, (B, L)),
            'ego_character': torch.randint(0, NUM_CHARACTERS, (B, L)),
            'opponent_character': torch.randint(0, NUM_CHARACTERS, (B, L)),
            'ego_action': torch.randint(0, NUM_ACTIONS, (B, L)),
            'opponent_action': torch.randint(0, NUM_ACTIONS, (B, L)),
            'gamestate': torch.randn(B, L, gamestate_dim),
            'controller': torch.randn(B, L, controller_dim),
        },
        batch_size=(B,),
    )
    print('inputs:')
    for k, v in inputs.items():
        print(k, v.shape)
    outputs = model(inputs)
    print('outputs:')
    for k, v in outputs.items():
        print(k, v.shape)
    # should be (B, L, out_dim) for each head
    print('Model test passed.')
