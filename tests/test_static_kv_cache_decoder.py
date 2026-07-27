import pytest
import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.decoder import (
    DecoderOnlyTransformer,
    StaticPastKeyValues,
)


def _build_model() -> DecoderOnlyTransformer:
    torch.manual_seed(42)

    model = DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            num_heads=2,
            num_layers=3,
            dropout=0.0,
        )
    )
    model.eval()
    return model


def test_allocate_static_cache_creates_one_cache_per_layer() -> None:
    model = _build_model()

    caches = model.allocate_static_cache(batch_size=2)

    assert len(caches) == 3

    for cache in caches:
        assert cache.key.shape == (2, 2, 8, 8)
        assert cache.value.shape == (2, 2, 8, 8)
        assert cache.valid_length == 0

    assert len({cache.key.data_ptr() for cache in caches}) == 3
    assert len({cache.value.data_ptr() for cache in caches}) == 3


def test_static_cache_prefill_and_decode_match_full_forward() -> None:
    model = _build_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        full_logits = model(input_ids)

        static_caches = model.allocate_static_cache(batch_size=1)
        prefill_logits, returned_caches = model(
            input_ids[:, :4],
            static_key_values=static_caches,
            use_cache=True,
        )
        decode_logits, returned_caches = model(
            input_ids[:, 4:],
            static_key_values=returned_caches,
            use_cache=True,
        )

    cached_logits = torch.cat(
        [prefill_logits, decode_logits],
        dim=1,
    )

    torch.testing.assert_close(
        cached_logits,
        full_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    assert returned_caches is static_caches
    assert all(cache.valid_length == 5 for cache in static_caches)


def test_static_cache_matches_dynamic_cache() -> None:
    model = _build_model()
    prefill_ids = torch.tensor([[1, 2, 3, 4]])
    decode_id = torch.tensor([[5]])

    with torch.no_grad():
        _, dynamic_caches = model(
            prefill_ids,
            use_cache=True,
        )
        dynamic_logits, _ = model(
            decode_id,
            past_key_values=dynamic_caches,
            use_cache=True,
        )

        static_caches = model.allocate_static_cache(batch_size=1)
        model(
            prefill_ids,
            static_key_values=static_caches,
            use_cache=True,
        )
        static_logits, _ = model(
            decode_id,
            static_key_values=static_caches,
            use_cache=True,
        )

    torch.testing.assert_close(
        static_logits,
        dynamic_logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_static_cache_reuses_original_storage() -> None:
    model = _build_model()
    caches = model.allocate_static_cache(batch_size=1)

    key_pointers = [cache.key.data_ptr() for cache in caches]
    value_pointers = [cache.value.data_ptr() for cache in caches]

    with torch.no_grad():
        _, returned_caches = model(
            torch.tensor([[1, 2, 3, 4]]),
            static_key_values=caches,
            use_cache=True,
        )
        _, returned_caches = model(
            torch.tensor([[5]]),
            static_key_values=returned_caches,
            use_cache=True,
        )

    assert returned_caches is caches
    assert [cache.key.data_ptr() for cache in caches] == key_pointers
    assert [cache.value.data_ptr() for cache in caches] == value_pointers


def test_decoder_rejects_both_cache_modes() -> None:
    model = _build_model()
    input_ids = torch.tensor([[1]])

    with torch.no_grad():
        _, dynamic_caches = model(
            input_ids,
            use_cache=True,
        )

    static_caches = model.allocate_static_cache(batch_size=1)

    with pytest.raises(ValueError, match="不能同时使用"):
        model(
            input_ids,
            past_key_values=dynamic_caches,
            static_key_values=static_caches,
            use_cache=True,
        )


def test_decoder_rejects_inconsistent_static_cache_lengths() -> None:
    model = _build_model()
    caches = model.allocate_static_cache(batch_size=1)
    caches[0].valid_length = 1

    with pytest.raises(ValueError, match="valid_length 必须一致"):
        model(
            torch.tensor([[1]]),
            static_key_values=caches,
            use_cache=True,
        )

    assert [cache.valid_length for cache in caches] == [1, 0, 0]


def test_capacity_failure_does_not_partially_advance_layers() -> None:
    model = _build_model()
    caches = model.allocate_static_cache(batch_size=1)

    with torch.no_grad():
        model(
            torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
            static_key_values=caches,
            use_cache=True,
        )

    short_cache = caches[-1]
    short_cache.key = short_cache.key[:, :, :7, :]
    short_cache.value = short_cache.value[:, :, :7, :]

    with pytest.raises(ValueError, match="capacity"):
        model(
            torch.tensor([[8]]),
            static_key_values=caches,
            use_cache=True,
        )

    assert all(cache.valid_length == 7 for cache in caches)


def test_static_cache_return_type_is_reusable_state() -> None:
    model = _build_model()
    caches = model.allocate_static_cache(batch_size=1)

    with torch.no_grad():
        _, cache_state = model(
            torch.tensor([[1, 2]]),
            static_key_values=caches,
            use_cache=True,
        )

    assert isinstance(cache_state, tuple)
    assert all(
        type(cache) is type(caches[0])
        for cache in cache_state
    )

    typed_state: StaticPastKeyValues = cache_state
    assert typed_state[0].valid_length == 2