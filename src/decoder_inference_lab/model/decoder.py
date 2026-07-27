from __future__ import annotations

import torch
from torch import nn

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.attention import KVCache
from decoder_inference_lab.model.block import TransformerBlock
from decoder_inference_lab.model.cache import StaticKVCache
from decoder_inference_lab.model.norm import RMSNorm

PastKeyValues = tuple[KVCache, ...]
StaticPastKeyValues = tuple[StaticKVCache, ...]
CacheState = PastKeyValues | StaticPastKeyValues


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config)
            for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(
            config.d_model,
            config.norm_eps,
        )
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def allocate_static_cache(
        self,
        batch_size: int,
    ) -> StaticPastKeyValues:
        """为每个 Transformer Layer 分配独立的 Static KV Cache。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")

        reference = self.token_embedding.weight

        return tuple(
            StaticKVCache.allocate(
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                max_seq_len=self.config.max_seq_len,
                head_dim=self.config.head_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
            for _ in self.blocks
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: PastKeyValues | None = None,
        static_key_values: StaticPastKeyValues | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, CacheState]:
        # input_ids: [B, Q]
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids 必须是 [B, T]，"
                f"实际 shape 为 {tuple(input_ids.shape)}"
            )

        if (
            past_key_values is not None
            and static_key_values is not None
        ):
            raise ValueError(
                "past_key_values 与 static_key_values 不能同时使用"
            )

        if static_key_values is not None and not use_cache:
            raise ValueError(
                "使用 static_key_values 时 use_cache 必须为 True"
            )

        if (
            past_key_values is not None
            and len(past_key_values) != len(self.blocks)
        ):
            raise ValueError(
                "past_key_values 的层数必须与 "
                "TransformerBlock 数量一致"
            )

        if (
            static_key_values is not None
            and len(static_key_values) != len(self.blocks)
        ):
            raise ValueError(
                "static_key_values 的层数必须与 "
                "TransformerBlock 数量一致"
            )

        batch_size, query_length = input_ids.shape

        if past_key_values is not None:
            first_key, _ = past_key_values[0]
            past_length = first_key.size(2)
        elif static_key_values is not None:
            layer_lengths = {
                layer_cache.valid_length
                for layer_cache in static_key_values
            }
            if len(layer_lengths) != 1:
                raise ValueError(
                    "所有层的 Static KV Cache valid_length 必须一致"
                )
            past_length = static_key_values[0].valid_length
        else:
            past_length = 0

        total_length = past_length + query_length

        if total_length > self.config.max_seq_len:
            raise ValueError(
                f"total_length={total_length} "
                f"超过 max_seq_len={self.config.max_seq_len}"
            )

        if static_key_values is not None:
            reference = self.token_embedding.weight

            for layer_cache in static_key_values:
                if (
                    layer_cache.key.ndim != 4
                    or layer_cache.value.shape
                    != layer_cache.key.shape
                ):
                    raise ValueError(
                        "Static KV Cache 必须具有一致的 "
                        "[B, H, capacity, Dh] shape"
                    )
                if total_length > layer_cache.capacity:
                    raise ValueError(
                        "Static KV Cache capacity 不足"
                    )
                if (
                    layer_cache.key.size(0) != batch_size
                    or layer_cache.key.size(1)
                    != self.config.num_heads
                    or layer_cache.key.size(3)
                    != self.config.head_dim
                ):
                    raise ValueError(
                        "Static KV Cache 的 batch、head "
                        "或 head_dim 与模型输入不一致"
                    )
                if (
                    layer_cache.key.device != input_ids.device
                    or layer_cache.value.device != input_ids.device
                ):
                    raise ValueError(
                        "Static KV Cache 与输入必须位于同一 device"
                    )
                if (
                    layer_cache.key.dtype != reference.dtype
                    or layer_cache.value.dtype != reference.dtype
                ):
                    raise ValueError(
                        "Static KV Cache dtype 与模型必须一致"
                    )

        positions = torch.arange(
            past_length,
            total_length,
            device=input_ids.device,
        )

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)
        hidden = token_vectors + position_vectors

        present_key_values: list[KVCache] = []

        for layer_index, block in enumerate(self.blocks):
            if past_key_values is None:
                layer_past = None
            else:
                layer_past = past_key_values[layer_index]

            if static_key_values is None:
                layer_static = None
            else:
                layer_static = static_key_values[layer_index]

            block_result = block(
                hidden,
                past_key_value=layer_past,
                static_cache=layer_static,
                use_cache=use_cache,
            )

            if use_cache:
                hidden, layer_present = block_result
                if static_key_values is None:
                    present_key_values.append(layer_present)
            else:
                hidden = block_result

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        if use_cache:
            if static_key_values is not None:
                return logits, static_key_values
            return logits, tuple(present_key_values)

        return logits