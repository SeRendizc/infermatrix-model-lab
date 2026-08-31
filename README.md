# Decoder Inference Lab

[中文](README.zh-CN.md) | **EN**

A learning project on Transformer inference. It starts with a small decoder-only model in PyTorch and adds training, greedy generation, and dynamic/static KV caches step by step.

The attention, normalization, feed-forward, and decoder modules can be read and tested separately. Experiments compare cached decoding with a full forward pass, examine cache allocation and updates, and measure prefill and decode time.

## Running the training example

Requires Python 3.10+ and PyTorch 2.12.1. In a virtual environment:

```bash
git clone https://github.com/SeRendizc/decoder-inference-lab.git
cd decoder-inference-lab
python -m pip install -e ".[dev]"
python scripts/train_tiny_corpus.py
```

This runs a two-layer model on a short repeated string for 200 training steps, then prints the loss, token accuracy, and a generated continuation. It runs on CPU and is a quick check of the training and generation paths. The model settings and training parameters are defined in [the script](scripts/train_tiny_corpus.py).

For CUDA, install the appropriate PyTorch wheel for your platform; the project does not select a CUDA package index automatically.

## Model

The decoder uses multi-head causal attention, RMSNorm, pre-norm residual blocks, an MLP, and learned token and position embeddings. Training uses next-token cross-entropy with a UTF-8 byte tokenizer.

```text
Token IDs -> token + position embeddings
          -> [RMSNorm -> attention -> residual
              RMSNorm -> MLP       -> residual] x layers
          -> RMSNorm -> LM head -> logits
```

Start with [attention.py](src/decoder_inference_lab/model/attention.py) and [decoder.py](src/decoder_inference_lab/model/decoder.py) for the forward pass. [generation.py](src/decoder_inference_lab/generation.py) handles greedy decoding, EOS, and the context window. The repository contains the implementation and small experiments; it does not ship pretrained weights.

## KV cache

Both cache implementations use the same model forward path:

- **Dynamic:** store each layer's keys and values and concatenate the new tokens during decoding.
- **Static:** allocate fixed-capacity tensors, write new keys and values in place, and track the initialized region with `valid_length`.

Generation selects the implementation with `cache_implementation="dynamic"` or `"static"`. It prefills the prompt, then passes one new token at a time. When the context window fills, it rebuilds the cache from the latest window.

The static cache lives in [cache.py](src/decoder_inference_lab/model/cache.py). Its tensors have shape `[batch, heads, capacity, head_dim]`; resetting the cache preserves the allocation.

To check the model/cache integration:

```bash
python -m pytest tests/test_static_kv_cache_decoder.py -q
```

The tests compare cached logits with a full forward pass, compare static and dynamic results, and check that storage is reused. They also cover inconsistent layer lengths and capacity failures. [Generation tests](tests/test_static_kv_cache_generation.py) cover the same behavior through the decoding loop.

## Experiments

Three scripts cover the current comparisons:

```bash
python scripts/profile_kv_cache.py
python scripts/profile_prefill_decode.py
python scripts/profile_cache_implementations.py
```

`profile_kv_cache.py` compares cached generation with full-context recomputation. `profile_prefill_decode.py` measures prompt prefill, one-token decode, and KV tensor size across context lengths. `profile_cache_implementations.py` compares dynamic and static generation after checking that their outputs agree.

The scripts use warmup, repeated timing, and CUDA synchronization. They choose CUDA when available and otherwise run on CPU; adjust the model sizes in the scripts for a smaller machine. The dynamic/static timings include cache allocation, and the reported KV tensor size excludes the model’s other GPU allocations. Keep the hardware, dtype, and sequence lengths with any results you publish.

The checked-in [environment report](artifacts/d1_environment.json) was captured on WSL2 with an RTX 3060 Laptop GPU and PyTorch 2.12.1+cu130. To inspect your own setup:

```bash
python scripts/verify_environment.py --check-compile --output artifacts/local_environment.json
```

## Tests and further experiments

```bash
python -m pytest tests -q
python -m ruff check .
```

Tests cover model shapes, gradients, causality, training, and cache behavior. CUDA-specific tests skip when CUDA is unavailable.

The next experiments are attention-kernel comparisons and profiler traces, followed by dtype and `torch.compile` measurements. The current scripts are local model experiments rather than a concurrent serving benchmark.

Related: [Agent Eval Lab](https://github.com/SeRendizc/agent-eval-lab) tests model-server APIs; [Agent Runtime Lab](https://github.com/SeRendizc/agent-runtime-lab) explores tool execution and recovery.
