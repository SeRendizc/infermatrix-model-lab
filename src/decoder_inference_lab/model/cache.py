from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StaticKVCache:
    """预先分配固定容量的单层 KV Cache。"""

    # 完整存储空间，shape 都是 [B, H, capacity, Dh]
    key: torch.Tensor
    value: torch.Tensor

    # 已经写入多少个 token，同时也是下次写入的起点
    valid_length: int = 0

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> StaticKVCache:
        """一次性分配固定容量的 K/V 存储空间。"""
        cache_shape = (
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
        )

        key = torch.empty(
            cache_shape,
            device=device,
            dtype=dtype,
        )
        value = torch.empty(
            cache_shape,
            device=device,
            dtype=dtype,
        )

        return cls(
            key=key,
            value=value,
        )

    @property
    def capacity(self) -> int:
        """Cache 在 token 维度上的最大容量。"""
        return self.key.size(2)

    def reset(self) -> None:
        """逻辑清空 Cache，同时保留已经分配的底层存储。"""
        self.valid_length = 0

    def update(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """写入本轮新产生的 K/V，并返回当前有效区域。"""
        if new_key.shape != new_value.shape:
            raise ValueError("new_key 与 new_value 的 shape 必须一致")

        if new_key.ndim != 4:
            raise ValueError("new_key/new_value 必须是 [B, H, Q, Dh]")

        expected_shape = (
            self.key.size(0),
            self.key.size(1),
            self.key.size(3),
        )
        actual_shape = (
            new_key.size(0),
            new_key.size(1),
            new_key.size(3),
        )

        if actual_shape != expected_shape:
            raise ValueError("new_key/new_value 与 Cache 的 batch、head 或 head_dim 不一致")

        if new_key.device != self.key.device or new_value.device != self.value.device:
            raise ValueError("new_key/new_value 与 Cache 必须位于同一 device")

        if new_key.dtype != self.key.dtype or new_value.dtype != self.value.dtype:
            raise ValueError("new_key/new_value 与 Cache 的 dtype 必须一致")

        query_length = new_key.size(2)

        if query_length == 0:
            raise ValueError("本次写入的 query_length 不能为 0")

        start_position = self.valid_length
        end_position = start_position + query_length

        if end_position > self.capacity:
            raise ValueError("Cache capacity exceeded")

        self.key[
            :,
            :,
            start_position:end_position,
            :,
        ].copy_(new_key)

        self.value[
            :,
            :,
            start_position:end_position,
            :,
        ].copy_(new_value)

        self.valid_length = end_position

        return (
            self.key[:, :, : self.valid_length, :],
            self.value[:, :, : self.valid_length, :],
        )