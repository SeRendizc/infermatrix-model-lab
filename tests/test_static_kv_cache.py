import pytest
import torch

from decoder_inference_lab.model.cache import StaticKVCache


def _allocate_cache(
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


def test_allocate_creates_fixed_capacity_storage() -> None:
    cache = _allocate_cache(max_seq_len=8)

    assert cache.key.shape == (1, 2, 8, 4)
    assert cache.value.shape == (1, 2, 8, 4)
    assert cache.capacity == 8
    assert cache.valid_length == 0
    assert cache.key.dtype == torch.float32
    assert cache.value.dtype == torch.float32


def test_prefill_and_decode_update_valid_region() -> None:
    torch.manual_seed(42)

    cache = _allocate_cache(max_seq_len=8)

    key_pointer = cache.key.data_ptr()
    value_pointer = cache.value.data_ptr()

    prefill_key = torch.randn(1, 2, 4, 4)
    prefill_value = torch.randn(1, 2, 4, 4)

    valid_key, valid_value = cache.update(
        prefill_key,
        prefill_value,
    )

    assert cache.valid_length == 4
    assert valid_key.shape == (1, 2, 4, 4)
    assert valid_value.shape == (1, 2, 4, 4)

    torch.testing.assert_close(
        valid_key,
        prefill_key,
    )
    torch.testing.assert_close(
        valid_value,
        prefill_value,
    )

    decode_key = torch.randn(1, 2, 1, 4)
    decode_value = torch.randn(1, 2, 1, 4)

    valid_key, valid_value = cache.update(
        decode_key,
        decode_value,
    )

    assert cache.valid_length == 5
    assert valid_key.shape == (1, 2, 5, 4)
    assert valid_value.shape == (1, 2, 5, 4)

    torch.testing.assert_close(
        valid_key[:, :, :4, :],
        prefill_key,
    )
    torch.testing.assert_close(
        valid_value[:, :, :4, :],
        prefill_value,
    )
    torch.testing.assert_close(
        valid_key[:, :, 4:5, :],
        decode_key,
    )
    torch.testing.assert_close(
        valid_value[:, :, 4:5, :],
        decode_value,
    )

    assert cache.key.data_ptr() == key_pointer
    assert cache.value.data_ptr() == value_pointer


def test_update_rejects_capacity_overflow() -> None:
    cache = _allocate_cache(max_seq_len=4)

    cache.update(
        torch.randn(1, 2, 4, 4),
        torch.randn(1, 2, 4, 4),
    )

    with pytest.raises(
        ValueError,
        match="Cache capacity exceeded",
    ):
        cache.update(
            torch.randn(1, 2, 1, 4),
            torch.randn(1, 2, 1, 4),
        )

    assert cache.valid_length == 4