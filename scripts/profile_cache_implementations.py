from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.generation import (
    CacheImplementation,
    generate_greedy,
)
from decoder_inference_lab.model.decoder import DecoderOnlyTransformer


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmup_runs: int = 2,
    measured_runs: int = 5,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup_runs):
            function()

        elapsed_times: list[float] = []

        for _ in range(measured_runs):
            _synchronize(device)
            start_time = time.perf_counter()

            function()

            _synchronize(device)
            elapsed_times.append(time.perf_counter() - start_time)

    return statistics.median(elapsed_times)


def run_generation(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    cache_implementation: CacheImplementation,
) -> torch.Tensor:
    return generate_greedy(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        cache_implementation=cache_implementation,
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
    model = DecoderOnlyTransformer(config).to(device)
    model.eval()

    prompt_length = 256
    max_new_tokens = 128
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, prompt_length),
        device=device,
    )

    dynamic_output = run_generation(
        model,
        input_ids,
        max_new_tokens,
        config.max_seq_len,
        "dynamic",
    )
    static_output = run_generation(
        model,
        input_ids,
        max_new_tokens,
        config.max_seq_len,
        "static",
    )

    if not torch.equal(dynamic_output, static_output):
        raise AssertionError(
            "Dynamic 与 Static Cache 的生成结果不一致"
        )

    dynamic_seconds = benchmark(
        function=lambda: run_generation(
            model,
            input_ids,
            max_new_tokens,
            config.max_seq_len,
            "dynamic",
        ),
        device=device,
    )
    static_seconds = benchmark(
        function=lambda: run_generation(
            model,
            input_ids,
            max_new_tokens,
            config.max_seq_len,
            "static",
        ),
        device=device,
    )

    print(f"device: {device}")
    print(f"prompt length: {prompt_length}")
    print(f"generated tokens: {max_new_tokens}")
    print()
    print(f"dynamic median time: {dynamic_seconds * 1000:.2f} ms")
    print(f"static median time:  {static_seconds * 1000:.2f} ms")
    print(
        "dynamic/static ratio: "
        f"{dynamic_seconds / static_seconds:.2f}x"
    )
    print()
    print(
        "说明：该结果包含整次生成中的 Cache 分配成本；"
        "它衡量端到端路径，不是孤立的单步 Decode kernel。"
    )


if __name__ == "__main__":
    main()
