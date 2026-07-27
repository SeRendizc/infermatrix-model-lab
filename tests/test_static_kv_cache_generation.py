from __future__ import annotations

import pytest
import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.generation import generate_greedy
from decoder_inference_lab.model.cache import StaticKVCache
from decoder_inference_lab.model.decoder import DecoderOnlyTransformer


class StaticIncrementModel:
    """记录 Static Cache 生命周期，并预测下一个 token = 当前 token + 1。"""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.call_input_lengths: list[int] = []
        self.cache_lengths_before_call: list[int] = []
        self.cache_key_pointers: list[int] = []

    def eval(self) -> StaticIncrementModel:
        return self

    def allocate_static_cache(
        self,
        batch_size: int,
    ) -> tuple[StaticKVCache, ...]:
        return (
            StaticKVCache.allocate(
                batch_size=batch_size,
                num_heads=1,
                max_seq_len=self.max_seq_len,
                head_dim=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        static_key_values: tuple[StaticKVCache, ...],
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[StaticKVCache, ...]]:
        if not use_cache:
            raise ValueError("测试模型要求 use_cache=True")

        cache = static_key_values[0]
        query_length = input_ids.size(1)

        self.call_input_lengths.append(query_length)
        self.cache_lengths_before_call.append(cache.valid_length)
        self.cache_key_pointers.append(cache.key.data_ptr())

        new_key = torch.zeros(
            input_ids.size(0),
            1,
            query_length,
            1,
        )
        cache.update(new_key, torch.zeros_like(new_key))

        logits = torch.full(
            (
                input_ids.size(0),
                query_length,
                self.vocab_size,
            ),
            fill_value=-1000.0,
        )
        next_token_ids = (input_ids + 1) % self.vocab_size
        logits.scatter_(
            dim=-1,
            index=next_token_ids.unsqueeze(-1),
            value=1000.0,
        )

        return logits, static_key_values


def _build_decoder() -> DecoderOnlyTransformer:
    torch.manual_seed(42)
    model = DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
        )
    )
    model.eval()
    return model


def test_static_generation_uses_prefill_then_single_token_decode() -> None:
    model = StaticIncrementModel(
        vocab_size=8,
        max_seq_len=4,
    )

    generated = generate_greedy(
        model,
        torch.tensor([[1, 2]]),
        max_new_tokens=3,
        max_seq_len=4,
        cache_implementation="static",
    )

    assert torch.equal(
        generated,
        torch.tensor([[1, 2, 3, 4, 5]]),
    )
    assert model.call_input_lengths == [2, 1, 1]
    assert model.cache_lengths_before_call == [0, 2, 3]
    assert len(set(model.cache_key_pointers)) == 1


def test_static_generation_resets_and_refills_full_cache() -> None:
    model = StaticIncrementModel(
        vocab_size=8,
        max_seq_len=4,
    )

    generated = generate_greedy(
        model,
        torch.tensor([[0, 1, 2, 3]]),
        max_new_tokens=4,
        max_seq_len=4,
        cache_implementation="static",
    )

    assert torch.equal(
        generated,
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
    )
    assert model.call_input_lengths == [4, 4, 4, 4]
    assert model.cache_lengths_before_call == [0, 0, 0, 0]
    assert len(set(model.cache_key_pointers)) == 1


def test_dynamic_and_static_generation_match() -> None:
    model = _build_decoder()
    input_ids = torch.tensor([[1, 2, 3]])

    dynamic_output = generate_greedy(
        model,
        input_ids,
        max_new_tokens=4,
        max_seq_len=8,
        cache_implementation="dynamic",
    )
    static_output = generate_greedy(
        model,
        input_ids,
        max_new_tokens=4,
        max_seq_len=8,
        cache_implementation="static",
    )

    assert torch.equal(static_output, dynamic_output)


def test_static_generation_rejects_model_without_allocator() -> None:
    model = torch.nn.Identity()

    with pytest.raises(
        TypeError,
        match="allocate_static_cache",
    ):
        generate_greedy(
            model,
            torch.tensor([[1, 2]]),
            max_new_tokens=1,
            max_seq_len=4,
            cache_implementation="static",
        )


def test_generation_rejects_unknown_cache_implementation() -> None:
    model = _build_decoder()

    with pytest.raises(
        ValueError,
        match="dynamic.*static",
    ):
        generate_greedy(
            model,
            torch.tensor([[1, 2]]),
            max_new_tokens=1,
            max_seq_len=8,
            cache_implementation="unknown",  # type: ignore[arg-type]
        )