from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

from decoder_inference_lab.config import ModelConfig
from decoder_inference_lab.model.decoder import DecoderOnlyTransformer


def _synchronize(device: torch.device) -> None:
    """等待设备上的计算真正完成。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_without_cache(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, int]:
    """每轮重新计算完整上下文的生成路径。"""
    generated = input_ids
    processed_tokens = 0

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = generated[:, -max_seq_len:]

            logits = model(context)

            processed_tokens += context.numel()

            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

    return generated, processed_tokens


def generate_with_cache(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, int]:
    """使用 Prefill 和单 Token Decode 的生成路径。"""
    generated = input_ids

    if max_new_tokens <= 0:
        return generated, 0

    with torch.inference_mode():
        # Prefill：一次处理完整 prompt。
        logits, past_key_values = model(
            input_ids,
            use_cache=True,
        )

        processed_tokens = input_ids.numel()

        for step in range(max_new_tokens):
            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            # 最后一个 token 已生成，无需再执行模型。
            if step == max_new_tokens - 1:
                break

            # Decode：只处理最新生成的一个 token。
            logits, past_key_values = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

            processed_tokens += next_token.numel()

    return generated, processed_tokens


def benchmark(
    function: Callable[[], tuple[torch.Tensor, int]],
    device: torch.device,
    warmup_runs: int = 2,
    measured_runs: int = 5,
) -> tuple[float, int]:
    """预热后重复测量，并返回耗时中位数。"""
    for _ in range(warmup_runs):
        function()

    _synchronize(device)

    elapsed_times: list[float] = []
    processed_tokens = 0

    for _ in range(measured_runs):
        _synchronize(device)
        start_time = time.perf_counter()

        _, processed_tokens = function()

        _synchronize(device)
        end_time = time.perf_counter()

        elapsed_times.append(end_time - start_time)

    median_seconds = statistics.median(elapsed_times)

    return median_seconds, processed_tokens


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    prompt_length = 256
    max_new_tokens = 128
    max_seq_len = 512
    config = ModelConfig(
        vocab_size=256,
        max_seq_len=max_seq_len,
        d_model=512,
        num_heads=8,
        num_layers=8,
        dropout=0.0,
    )

    model = DecoderOnlyTransformer(config)
    model = model.to(device)
    model.eval()

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, prompt_length),
        device=device,
    )

    uncached_output, uncached_tokens = generate_without_cache(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
    )

    cached_output, cached_tokens = generate_with_cache(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
    )

    if not torch.equal(uncached_output, cached_output):
        raise AssertionError(
            "无 Cache 与有 Cache 的生成结果不一致"
        )

    uncached_seconds, uncached_tokens = benchmark(
        function=lambda: generate_without_cache(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
        ),
        device=device,
    )

    cached_seconds, cached_tokens = benchmark(
        function=lambda: generate_with_cache(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        ),
        device=device,
    )

    token_reduction = uncached_tokens / cached_tokens
    speedup = uncached_seconds / cached_seconds

    print(f"device: {device}")
    print(f"prompt length: {prompt_length}")
    print(f"generated tokens: {max_new_tokens}")
    print()
    print(f"without cache processed tokens: {uncached_tokens}")
    print(f"with cache processed tokens:    {cached_tokens}")
    print(f"token computation reduction:    {token_reduction:.2f}x")
    print()
    print(f"without cache median time: {uncached_seconds * 1000:.2f} ms")
    print(f"with cache median time:    {cached_seconds * 1000:.2f} ms")
    print(f"measured speedup:          {speedup:.2f}x")


if __name__ == "__main__":
    main()