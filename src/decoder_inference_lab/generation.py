from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from decoder_inference_lab.model.decoder import StaticPastKeyValues

CacheImplementation = Literal["dynamic", "static"]


def _reset_static_cache(
    static_key_values: StaticPastKeyValues,
) -> None:
    for layer_cache in static_key_values:
        layer_cache.reset()


def generate_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_id: int | None = None,
    cache_implementation: CacheImplementation = "dynamic",
) -> torch.Tensor:
    # input_ids: [B, T]
    if cache_implementation not in {"dynamic", "static"}:
        raise ValueError(
            "cache_implementation 必须是 'dynamic' 或 'static'"
        )

    generated = input_ids

    if max_new_tokens <= 0:
        return generated

    model.eval()

    with torch.no_grad():
        # Prefill：一次性处理当前完整上下文，并建立 KV Cache。
        context = generated[:, -max_seq_len:]

        if cache_implementation == "static":
            allocate_static_cache = getattr(
                model,
                "allocate_static_cache",
                None,
            )
            if allocate_static_cache is None:
                raise TypeError(
                    "Static Cache 模式要求模型实现 "
                    "allocate_static_cache(batch_size)"
                )

            static_key_values = allocate_static_cache(
                batch_size=context.size(0),
            )
            logits, cache_state = model(
                context,
                static_key_values=static_key_values,
                use_cache=True,
            )
        else:
            logits, cache_state = model(
                context,
                use_cache=True,
            )

        for step in range(max_new_tokens):
            # 使用最后一个位置的 logits 预测下一个 token。
            next_token_logits = logits[:, -1, :]

            next_token = next_token_logits.argmax(
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if (
                eos_token_id is not None
                and torch.all(next_token == eos_token_id)
            ):
                break

            # 最后一个 token 已经生成，不需要再进行一次模型前向。
            if step == max_new_tokens - 1:
                break

            if cache_implementation == "static":
                cache_length = cache_state[0].valid_length
            else:
                first_key, _ = cache_state[0]
                cache_length = first_key.size(2)

            if cache_length < max_seq_len:
                # Decode：Cache 未满，只输入刚生成的一个 token。
                if cache_implementation == "static":
                    logits, cache_state = model(
                        next_token,
                        static_key_values=cache_state,
                        use_cache=True,
                    )
                else:
                    logits, cache_state = model(
                        next_token,
                        past_key_values=cache_state,
                        use_cache=True,
                    )
            else:
                # Cache 已满：截取最新窗口，重新执行 Prefill。
                context = generated[:, -max_seq_len:]

                if cache_implementation == "static":
                    _reset_static_cache(cache_state)
                    logits, cache_state = model(
                        context,
                        static_key_values=cache_state,
                        use_cache=True,
                    )
                else:
                    logits, cache_state = model(
                        context,
                        use_cache=True,
                    )

    return generated