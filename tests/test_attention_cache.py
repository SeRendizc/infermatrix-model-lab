import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.attention import CausalSelfAttention


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


def test_attention_returns_complete_cache() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(_config()).eval()
    x = torch.randn(2, 4, 64)

    with torch.no_grad():
        output, cache = attention(
            x,
            use_cache=True,
        )

    key, value = cache

    assert output.shape == (2, 4, 64)
    assert key.shape == (2, 4, 4, 16)
    assert value.shape == (2, 4, 4, 16)


def test_single_token_cached_decode_matches_full_attention() -> None:
    torch.manual_seed(1)
    attention = CausalSelfAttention(_config()).eval()
    x = torch.randn(1, 5, 64)

    with torch.no_grad():
        full_output = attention(x)

        _, cache = attention(
            x[:, :4],
            use_cache=True,
        )
        decode_output, updated_cache = attention(
            x[:, 4:],
            past_key_value=cache,
            use_cache=True,
        )

    torch.testing.assert_close(
        decode_output,
        full_output[:, 4:],
        rtol=1e-5,
        atol=1e-6,
    )

    updated_key, updated_value = updated_cache

    assert updated_key.shape == (1, 4, 5, 16)
    assert updated_value.shape == (1, 4, 5, 16)


def test_multi_token_cached_decode_matches_full_attention() -> None:
    torch.manual_seed(2)
    attention = CausalSelfAttention(_config()).eval()
    x = torch.randn(1, 5, 64)

    with torch.no_grad():
        full_output = attention(x)

        _, cache = attention(
            x[:, :3],
            use_cache=True,
        )
        chunk_output, updated_cache = attention(
            x[:, 3:],
            past_key_value=cache,
            use_cache=True,
        )

    torch.testing.assert_close(
        chunk_output,
        full_output[:, 3:],
        rtol=1e-5,
        atol=1e-6,
    )

    updated_key, updated_value = updated_cache

    assert updated_key.shape == (1, 4, 5, 16)
    assert updated_value.shape == (1, 4, 5, 16)
