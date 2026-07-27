import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.decoder import DecoderOnlyTransformer


def _build_model() -> DecoderOnlyTransformer:
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        d_model=32,
        num_heads=4,
        num_layers=3,
        dropout=0.0,
    )

    model = DecoderOnlyTransformer(config)
    model.eval()

    return model


def test_decoder_cached_forward_matches_full_forward() -> None:
    model = _build_model()

    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5]],
        dtype=torch.long,
    )

    with torch.no_grad():
        # 对照组：一次处理全部 5 个 token。
        full_logits = model(input_ids)

        # Prefill：先处理前 4 个 token，并建立 Cache。
        prefill_logits, past_key_values = model(
            input_ids[:, :4],
            use_cache=True,
        )

        # Decode：只输入第 5 个 token，同时使用前 4 个 token 的 Cache。
        decode_logits, present_key_values = model(
            input_ids[:, 4:],
            past_key_values=past_key_values,
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

    assert len(past_key_values) == 3
    assert len(present_key_values) == 3


def test_decoder_cache_has_one_entry_per_layer() -> None:
    model = _build_model()

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        _, past_key_values = model(
            input_ids,
            use_cache=True,
        )

    assert len(past_key_values) == 3

    for key, value in past_key_values:
        assert key.shape == (2, 4, 4, 8)
        assert value.shape == (2, 4, 4, 8)


def test_decoder_cache_grows_during_decode() -> None:
    model = _build_model()

    prefill_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )
    next_token = torch.tensor(
        [[5]],
        dtype=torch.long,
    )

    with torch.no_grad():
        _, past_key_values = model(
            prefill_ids,
            use_cache=True,
        )

        _, present_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )

    for layer_past, layer_present in zip(
        past_key_values,
        present_key_values,
        strict=True,
    ):
        past_key, past_value = layer_past
        present_key, present_value = layer_present

        assert past_key.size(2) == 4
        assert past_value.size(2) == 4

        assert present_key.size(2) == 5
        assert present_value.size(2) == 5