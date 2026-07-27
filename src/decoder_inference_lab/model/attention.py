from __future__ import annotations

import math

import torch
from torch import nn

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.cache import StaticKVCache

KVCache = tuple[torch.Tensor, torch.Tensor]


def _build_causal_mask(
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = past_length + torch.arange(
        query_length,
        device=device,
    )
    key_positions = torch.arange(
        key_length,
        device=device,
    )

    query_positions = query_positions.unsqueeze(1)
    key_positions = key_positions.unsqueeze(0)

    causal_mask = key_positions <= query_positions

    return causal_mask


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        static_cache: StaticKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        # x: [B, Q, D]
        batch_size, query_length, _ = x.shape

        query = self.q_proj(x)
        new_key = self.k_proj(x)
        new_value = self.v_proj(x)

        query = query.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        new_key = new_key.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        new_value = new_value.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # query/new_key/new_value: [B, H, Q, Dₕ]

        if (
            past_key_value is not None
            and static_cache is not None
        ):
            raise ValueError("past_key_value 与 static_cache 不能同时使用")

        if static_cache is not None:
            if not use_cache:
                raise ValueError("使用 static_cache 时 use_cache 必须为 True")

            # 必须在 update() 推进 valid_length 之前读取
            past_length = static_cache.valid_length  # ①

            # 写入本轮 new K/V，取得全部有效 K/V
            key, value = static_cache.update(new_key, new_value)  # ②

        elif past_key_value is None:
            past_length = 0
            key = new_key
            value = new_value

        else:
            past_key, past_value = past_key_value
            past_length = past_key.shape[2]

            key = torch.cat(
                [past_key, new_key],
                dim=2,
            )
            value = torch.cat(
                [past_value, new_value],
                dim=2,
            )

        # key/value: [B, H, K, Dₕ]
        key_length = key.shape[2]

        causal_mask = _build_causal_mask(
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            device=x.device,
        )

        # causal_mask: [1, 1, Q, K]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        scores = query @ key.transpose(-2, -1)
        # scores: [B, H, Q, K]
        scores = scores / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        weights = torch.softmax(scores, dim=-1)

        context = weights @ value
        # context: [B, H, Q, Dₕ]

        context = context.transpose(1, 2)
        context = context.contiguous().view(
            batch_size,
            query_length,
            self.num_heads * self.head_dim,
        )

        output = self.out_proj(context)
        # output: [B, Q, D]

        if use_cache:
            present_key_value = (
                key,
                value,
            )
            return output, present_key_value

        return output
