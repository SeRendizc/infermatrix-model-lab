from __future__ import annotations

import torch
from torch import nn


def generate_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    # input_ids: [B, T]
    generated = input_ids

    if max_new_tokens <= 0:
        return generated

    model.eval()

    with torch.no_grad():
        # Prefill：一次性处理当前完整上下文，并建立 KV Cache。
        context = generated[:, -max_seq_len:]

        logits, past_key_values = model(
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

            first_key, _ = past_key_values[0]
            cache_length = first_key.size(2)

            if cache_length < max_seq_len:
                # Decode：Cache 未满，只输入刚生成的一个 token。
                logits, past_key_values = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # Cache 已满：截取最新窗口，重新执行 Prefill。
                context = generated[:, -max_seq_len:]

                logits, past_key_values = model(
                    context,
                    use_cache=True,
                )

    return generated