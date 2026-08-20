# Causal Self-Attention B+ Implementation Plan

> Historical note: this document predates the rename from InferMatrix Model Lab to Decoder Inference Lab.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Core Attention implementation remains user-owned by explicit agreement.

**Goal:** 为用户亲手实现 `CausalSelfAttention` 准备精确接口、失败测试、shape 路线和验证闭环，同时避免 Codex 提前生成核心答案。

**Architecture:** `attention.py` 只提供稳定类接口并在 `forward()` 中明确抛出 `NotImplementedError`；`test_attention.py` 从输出 shape 和 causal behavior 两个外部行为验证实现。用户完成首次实现后，再加入 backward 测试并进行逐行 review。

**Tech Stack:** Python 3.10、PyTorch 2.12.1、pytest、Ruff、WSL2 CUDA 环境。

---

### Task 1: PyCharm 文件忽略规则

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 在根目录忽略 PyCharm 工程文件**

在 `.gitignore` 的缓存规则后加入：

```gitignore
.idea/
```

- [ ] **Step 2: 验证规则，不改动 Git index**

Run:

```powershell
git check-ignore -v --no-index .idea/misc.xml
git status --short
```

Expected: 第一条命令指出根目录 `.gitignore` 中的 `.idea/`；已经进入 index 的 `.idea` 文件仍可能显示为 `A`。本任务不执行 `git rm --cached`、不删除文件、不 commit。

### Task 2: 创建第一条 RED shape 测试

**Files:**
- Create: `tests/test_attention.py`

- [ ] **Step 1: 写入 shape 行为测试**

```python
import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.model.attention import CausalSelfAttention


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=16,
        d_model=64,
        num_heads=4,
        num_layers=1,
    )


def test_attention_preserves_shape() -> None:
    attention = CausalSelfAttention(_config())
    x = torch.randn(2, 5, 64)

    output = attention(x)

    assert output.shape == (2, 5, 64)
```

- [ ] **Step 2: 运行测试并确认首次失败原因**

Run:

```bash
/root/.venvs/infermatrix-model-lab/bin/python -m pytest \
  /mnt/d/infermatrix_model_lab/tests/test_attention.py::test_attention_preserves_shape -v
```

Expected: collection 因 `infermatrix_model_lab.model.attention` 尚不存在而失败。这证明测试确实覆盖新接口。

### Task 3: 创建 B+ 接口骨架并保持 RED

**Files:**
- Create: `src/infermatrix_model_lab/model/attention.py`

- [ ] **Step 1: 写入不含核心答案的稳定接口**

```python
from __future__ import annotations

import torch
from torch import nn

from infermatrix_model_lab.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        raise NotImplementedError("请按 B+ 路线亲手实现 causal self-attention")
```

- [ ] **Step 2: 再次运行 shape 测试**

Run: 与 Task 2 Step 2 相同。

Expected: 测试被正常收集，并在调用 `forward()` 时因上述 `NotImplementedError` 失败。此时 RED 原因已从“模块缺失”推进到“用户核心实现缺失”。

### Task 4: 添加 causal behavior RED 测试

**Files:**
- Modify: `tests/test_attention.py`

- [ ] **Step 1: 加入不窥视内部实现的 causal 测试**

```python
def test_future_tokens_do_not_change_past_outputs() -> None:
    torch.manual_seed(7)
    attention = CausalSelfAttention(_config()).eval()

    shared_prefix = torch.randn(1, 3, 64)
    first_future = torch.randn(1, 2, 64)
    second_future = torch.randn(1, 2, 64) + 10.0

    first_input = torch.cat([shared_prefix, first_future], dim=1)
    second_input = torch.cat([shared_prefix, second_future], dim=1)

    first_output = attention(first_input)
    second_output = attention(second_input)

    torch.testing.assert_close(
        first_output[:, :3],
        second_output[:, :3],
        rtol=0.0,
        atol=1e-6,
    )
```

- [ ] **Step 2: 运行 attention 测试文件**

Run:

```bash
/root/.venvs/infermatrix-model-lab/bin/python -m pytest \
  /mnt/d/infermatrix_model_lab/tests/test_attention.py -v
```

Expected: 两项测试均因用户实现尚未完成而失败。测试不读取 attention weights，不要求测试专用生产接口。

### Task 5: 用户完成首次核心实现

**Files:**
- Modify by user: `src/infermatrix_model_lab/model/attention.py`

- [ ] **Step 1: 按固定 shape 契约实现**

实现必须满足以下逐步契约；这是路线卡，不提供可复制的完整答案：

```text
x                    [B,T,D]
Q/K/V projection     [B,T,D]
split heads          [B,H,T,Dh]
attention scores     [B,H,T,T]
causal softmax       [B,H,T,T]
weighted values      [B,H,T,Dh]
merge heads          [B,T,D]
output projection    [B,T,D]
```

固定约束：缩放因子使用 `sqrt(Dh)`；mask 必须由当前 `T` 和 `x.device` 得到；不硬编码 CUDA；合并 head 前处理 transpose 后的非连续内存。

- [ ] **Step 2: 用户运行两个测试并把输出交给 Codex**

Run: Task 4 Step 2 的命令。

Expected: 两项测试 PASS。如果失败，用户先提供 traceback 和本人对失败 shape 的判断，Codex 再做局部提示。

### Task 6: 首次实现后的 backward 验收

**Files:**
- Modify after first implementation: `tests/test_attention.py`

- [ ] **Step 1: 由 Codex 在用户首次实现后加入梯度测试**

```python
def test_attention_supports_backward() -> None:
    attention = CausalSelfAttention(_config())
    x = torch.randn(2, 5, 64, requires_grad=True)

    attention(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    parameter_gradients = [
        parameter.grad for parameter in attention.parameters() if parameter.requires_grad
    ]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
```

- [ ] **Step 2: 全量验证**

Run:

```bash
/root/.venvs/infermatrix-model-lab/bin/python -m pytest \
  /mnt/d/infermatrix_model_lab/tests -q
/root/.venvs/infermatrix-model-lab/bin/python -m ruff check \
  /mnt/d/infermatrix_model_lab
```

Expected: 全部 pytest 与 Ruff 通过。Codex 随后 review shape、mask、device、数值稳定性和可读性；不自动 commit。
