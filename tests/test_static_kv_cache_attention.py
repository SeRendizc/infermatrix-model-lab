from types import SimpleNamespace
from typing import cast

import pytest
import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.attention import (
    CausalSelfAttention,
)
from decoder_inference_lab.model.cache import StaticKVCache


def _build_attention() -> CausalSelfAttention:
    config = cast(
        ModelConfig,
        SimpleNamespace(
            d_model=8,
            num_heads=2,
            head_dim=4,
        ),
    )
    return CausalSelfAttention(config)


def _allocate_static_cache(
    max_seq_len: int = 8,
) -> StaticKVCache:
    return StaticKVCache.allocate(
        batch_size=1,
        num_heads=2,
        max_seq_len=max_seq_len,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_static_cache_prefill_matches_regular_attention() -> None:
    torch.manual_seed(42)

    attention = _build_attention()
    x = torch.randn(1, 4, 8)

    regular_output = attention(x)

    static_cache = _allocate_static_cache()

    static_output, present_key_value = attention(
        x,
        static_cache=static_cache,
        use_cache=True,
    )

    torch.testing.assert_close(
        static_output,
        regular_output,
    )

    assert static_cache.valid_length == 4
    assert present_key_value[0].shape == (1, 2, 4, 4)
    assert present_key_value[1].shape == (1, 2, 4, 4)


def test_static_cache_decode_matches_dynamic_cache() -> None:
    torch.manual_seed(42)

    attention = _build_attention()

    prefill_input = torch.randn(1, 4, 8)
    decode_input = torch.randn(1, 1, 8)

    dynamic_prefill_output, dynamic_cache = attention(
        prefill_input,
        use_cache=True,
    )
    dynamic_decode_output, dynamic_cache = attention(
        decode_input,
        past_key_value=dynamic_cache,
        use_cache=True,
    )

    static_cache = _allocate_static_cache()

    static_prefill_output, _ = attention(
        prefill_input,
        static_cache=static_cache,
        use_cache=True,
    )
    static_decode_output, static_present = attention(
        decode_input,
        static_cache=static_cache,
        use_cache=True,
    )

    torch.testing.assert_close(
        static_prefill_output,
        dynamic_prefill_output,
    )
    torch.testing.assert_close(
        static_decode_output,
        dynamic_decode_output,
    )
    torch.testing.assert_close(
        static_present[0],
        dynamic_cache[0],
    )
    torch.testing.assert_close(
        static_present[1],
        dynamic_cache[1],
    )

    assert static_cache.valid_length == 5


def test_static_cache_keeps_storage_during_decode() -> None:
    torch.manual_seed(42)

    attention = _build_attention()
    static_cache = _allocate_static_cache()

    key_pointer = static_cache.key.data_ptr()
    value_pointer = static_cache.value.data_ptr()

    attention(
        torch.randn(1, 4, 8),
        static_cache=static_cache,
        use_cache=True,
    )
    attention(
        torch.randn(1, 1, 8),
        static_cache=static_cache,
        use_cache=True,
    )

    assert static_cache.valid_length == 5
    assert static_cache.key.data_ptr() == key_pointer
    assert static_cache.value.data_ptr() == value_pointer


def test_attention_rejects_dynamic_and_static_cache_together() -> None:
    attention = _build_attention()
    static_cache = _allocate_static_cache()

    past_key_value = (
        torch.randn(1, 2, 2, 4),
        torch.randn(1, 2, 2, 4),
    )

    with pytest.raises(
        ValueError,
        match="不能同时使用",
    ):
        attention(
            torch.randn(1, 1, 8),
            past_key_value=past_key_value,
            static_cache=static_cache,
            use_cache=True,
        )

    assert static_cache.valid_length == 0


def test_static_cache_requires_use_cache() -> None:
    attention = _build_attention()
    static_cache = _allocate_static_cache()

    with pytest.raises(
        ValueError,
        match="use_cache 必须为 True",
    ):
        attention(
            torch.randn(1, 1, 8),
            static_cache=static_cache,
            use_cache=False,
        )

    assert static_cache.valid_length == 0