import torch
from torch import nn

from decoder_inference_lab.generation import generate_greedy

KVCache = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[KVCache, ...]


class CachedIncrementModel(nn.Module):
    """预测下一个 token 为当前 token + 1 的缓存模型。"""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # 记录每次前向输入了几个 token。
        self.call_input_lengths: list[int] = []

        # 记录每次前向是否携带历史 Cache。
        self.call_used_past: list[bool] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, PastKeyValues]:
        # input_ids: [B, T]
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError("输入超过 max_seq_len")

        self.call_input_lengths.append(input_ids.size(1))
        self.call_used_past.append(past_key_values is not None)

        batch_size, query_length = input_ids.shape

        logits = torch.full(
            (
                batch_size,
                query_length,
                self.vocab_size,
            ),
            fill_value=-1000.0,
            device=input_ids.device,
        )

        next_token_ids = (input_ids + 1) % self.vocab_size

        logits.scatter_(
            dim=-1,
            index=next_token_ids.unsqueeze(-1),
            value=1000.0,
        )

        if not use_cache:
            return logits

        if past_key_values is None:
            past_length = 0
        else:
            past_key, _ = past_key_values[0]
            past_length = past_key.size(2)

        total_length = past_length + query_length

        key = torch.zeros(
            batch_size,
            1,
            total_length,
            1,
            device=input_ids.device,
        )
        value = torch.zeros_like(key)

        present_key_values: PastKeyValues = ((key, value),)

        return logits, present_key_values


class CachedEosModel(nn.Module):
    """始终预测 EOS 的缓存模型。"""

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, PastKeyValues]:
        batch_size, query_length = input_ids.shape

        logits = torch.full(
            (
                batch_size,
                query_length,
                self.vocab_size,
            ),
            fill_value=-1000.0,
            device=input_ids.device,
        )

        logits[:, -1, self.eos_token_id] = 1000.0

        if not use_cache:
            return logits

        if past_key_values is None:
            past_length = 0
        else:
            past_key, _ = past_key_values[0]
            past_length = past_key.size(2)

        total_length = past_length + query_length

        key = torch.zeros(
            batch_size,
            1,
            total_length,
            1,
            device=input_ids.device,
        )
        value = torch.zeros_like(key)

        present_key_values: PastKeyValues = ((key, value),)

        return logits, present_key_values


def test_generate_greedy_appends_predicted_tokens() -> None:
    model = CachedIncrementModel(
        vocab_size=8,
        max_seq_len=4,
    )
    input_ids = torch.tensor([[1, 2]])

    generated = generate_greedy(
        model,
        input_ids,
        max_new_tokens=3,
        max_seq_len=4,
    )

    expected = torch.tensor([[1, 2, 3, 4, 5]])

    assert torch.equal(generated, expected)


def test_generate_greedy_uses_prefill_then_decode() -> None:
    model = CachedIncrementModel(
        vocab_size=8,
        max_seq_len=4,
    )
    input_ids = torch.tensor([[1, 2]])

    generate_greedy(
        model,
        input_ids,
        max_new_tokens=3,
        max_seq_len=4,
    )

    assert model.call_input_lengths == [2, 1, 1]
    assert model.call_used_past == [False, True, True]


def test_generate_greedy_refills_when_cache_is_full() -> None:
    model = CachedIncrementModel(
        vocab_size=8,
        max_seq_len=4,
    )
    input_ids = torch.tensor([[0, 1, 2, 3]])

    generated = generate_greedy(
        model,
        input_ids,
        max_new_tokens=4,
        max_seq_len=4,
    )

    expected = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7]]
    )

    assert torch.equal(generated, expected)

    # Cache 从一开始就是满的，因此每次都用最新窗口重新 Prefill。
    assert model.call_input_lengths == [4, 4, 4, 4]
    assert model.call_used_past == [False, False, False, False]


def test_generate_greedy_stops_after_eos() -> None:
    eos_token_id = 7

    model = CachedEosModel(
        vocab_size=8,
        eos_token_id=eos_token_id,
    )
    input_ids = torch.tensor([[1, 2]])

    generated = generate_greedy(
        model,
        input_ids,
        max_new_tokens=10,
        max_seq_len=4,
        eos_token_id=eos_token_id,
    )

    expected = torch.tensor([[1, 2, 7]])

    assert torch.equal(generated, expected)


def test_generate_greedy_generates_nothing_when_limit_is_zero() -> None:
    model = CachedIncrementModel(
        vocab_size=8,
        max_seq_len=4,
    )
    input_ids = torch.tensor([[1, 2]])

    generated = generate_greedy(
        model,
        input_ids,
        max_new_tokens=0,
        max_seq_len=4,
    )

    assert torch.equal(generated, input_ids)
    assert model.call_input_lengths == []