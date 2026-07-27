import torch
from torch import nn

from decoder_inference_lab.model.attention import KVCache
from decoder_inference_lab.model.block import TransformerBlock
from decoder_inference_lab.model.cache import StaticKVCache


class RecordingAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_static_cache: StaticKVCache | None = None

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        static_cache: StaticKVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        self.received_static_cache = static_cache

        attention_delta = torch.ones_like(x)

        if use_cache:
            if static_cache is None:
                raise ValueError("测试要求传入 static_cache")

            batch_size, query_length, _ = x.shape

            new_key = torch.zeros(
                batch_size,
                2,
                query_length,
                4,
                dtype=x.dtype,
                device=x.device,
            )
            new_value = torch.zeros_like(new_key)

            present_key_value = static_cache.update(
                new_key,
                new_value,
            )

            return attention_delta, present_key_value

        return attention_delta


class ZeroMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _build_test_block() -> TransformerBlock:
    block = TransformerBlock.__new__(TransformerBlock)
    nn.Module.__init__(block)

    block.norm_1 = nn.Identity()
    block.attention = RecordingAttention()
    block.norm_2 = nn.Identity()
    block.mlp = ZeroMLP()

    return block


def _allocate_static_cache() -> StaticKVCache:
    return StaticKVCache.allocate(
        batch_size=1,
        num_heads=2,
        max_seq_len=8,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_block_passes_static_cache_to_attention() -> None:
    block = _build_test_block()
    static_cache = _allocate_static_cache()

    x = torch.zeros(1, 3, 8)

    output, present_key_value = block(
        x,
        static_cache=static_cache,
        use_cache=True,
    )

    assert block.attention.received_static_cache is static_cache
    assert static_cache.valid_length == 3

    torch.testing.assert_close(
        output,
        torch.ones_like(x),
    )

    assert present_key_value[0].shape == (1, 2, 3, 4)
    assert present_key_value[1].shape == (1, 2, 3, 4)


def test_block_without_cache_preserves_original_path() -> None:
    block = _build_test_block()

    x = torch.zeros(1, 3, 8)

    output = block(
        x,
        use_cache=False,
    )

    assert isinstance(output, torch.Tensor)

    torch.testing.assert_close(
        output,
        torch.ones_like(x),
    )