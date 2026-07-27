from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import TypeAlias

import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.decoder import DecoderOnlyTransformer

KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]
PastKeyValues: TypeAlias = tuple[KVCache, ...]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_call(
    function: Callable[[], object],
    device: torch.device,
    warmup_runs: int = 5,
    measured_runs: int = 20,
    calls_per_run: int = 10,
) -> float:
    """返回单次函数调用耗时的中位数，单位为秒。"""
    with torch.inference_mode():
        for _ in range(warmup_runs):
            for _ in range(calls_per_run):
                function()

        _synchronize(device)

        per_call_times: list[float] = []

        for _ in range(measured_runs):
            _synchronize(device)
            start_time = time.perf_counter()

            for _ in range(calls_per_run):
                function()

            _synchronize(device)
            end_time = time.perf_counter()

            elapsed_seconds = end_time - start_time
            per_call_times.append(
                elapsed_seconds / calls_per_run
            )

    return statistics.median(per_call_times)


def cache_size_bytes(
    past_key_values: PastKeyValues,
) -> int:
    """计算当前各层 K/V Cache 张量占用的总字节数。"""
    total_bytes = 0

    for key, value in past_key_values:
        total_bytes += key.numel() * key.element_size()
        total_bytes += value.numel() * value.element_size()

    return total_bytes


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def profile_context_length(
    model: DecoderOnlyTransformer,
    config: ModelConfig,
    context_length: int,
    device: torch.device,
) -> tuple[float, float, float]:
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, context_length),
        device=device,
    )

    next_token = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 1),
        device=device,
    )

    def run_prefill() -> object:
        return model(
            input_ids,
            use_cache=True,
        )

    prefill_seconds = benchmark_call(
        function=run_prefill,
        device=device,
    )

    with torch.inference_mode():
        _, past_key_values = model(
            input_ids,
            use_cache=True,
        )

    def run_decode() -> object:
        # 每次复用相同的 past_key_values，
        # 这样测量的是固定上下文长度下的单 Token Decode。
        return model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )

    decode_seconds = benchmark_call(
        function=run_decode,
        device=device,
    )

    cache_mib = bytes_to_mib(
        cache_size_bytes(past_key_values)
    )

    return (
        prefill_seconds * 1000,
        decode_seconds * 1000,
        cache_mib,
    )


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = ModelConfig(
        vocab_size=256,
        max_seq_len=512,
        d_model=512,
        num_heads=8,
        num_layers=8,
        dropout=0.0,
    )

    model = DecoderOnlyTransformer(config)
    model = model.to(device)
    model.eval()

    context_lengths = [32, 64, 128, 256, 384]

    print(f"device: {device}")
    print(f"dtype: {next(model.parameters()).dtype}")
    print(f"layers: {config.num_layers}")
    print(f"d_model: {config.d_model}")
    print()
    print(
        f"{'context':>8} "
        f"{'prefill ms':>12} "
        f"{'decode ms':>12} "
        f"{'cache MiB':>12}"
    )
    print("-" * 50)

    for context_length in context_lengths:
        prefill_ms, decode_ms, cache_mib = (
            profile_context_length(
                model=model,
                config=config,
                context_length=context_length,
                device=device,
            )
        )

        print(
            f"{context_length:>8} "
            f"{prefill_ms:>12.3f} "
            f"{decode_ms:>12.3f} "
            f"{cache_mib:>12.3f}"
        )


if __name__ == "__main__":
    main()