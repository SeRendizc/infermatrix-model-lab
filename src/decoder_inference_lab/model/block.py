from __future__ import annotations

import torch
from torch import nn

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.attention import CausalSelfAttention, KVCache
from decoder_inference_lab.model.cache import StaticKVCache
from decoder_inference_lab.model.mlp import FeedForward
from decoder_inference_lab.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        static_cache: StaticKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        # x: [B, T, D]
        attention_input = self.norm_1(x)

        attention_result = self.attention(
            attention_input,
            past_key_value=past_key_value,
            static_cache=static_cache,
            use_cache=use_cache,
        )

        attention_delta: torch.Tensor
        present_key_value: KVCache | None = None

        if use_cache:
            attention_delta, present_key_value = attention_result
        else:
            attention_delta = attention_result

        x = x + attention_delta

        mlp_input = self.norm_2(x)
        mlp_delta = self.mlp(mlp_input)

        x = x + mlp_delta

        if present_key_value is not None:
            return x, present_key_value

        return x
