# Decoder Inference Lab

**中文** | [EN](README.md)

这是一个学习Transformer推理机制的实验项目。从用PyTorch实现小型Decoder-only模型开始，逐步加入训练、贪心生成和动态/静态KV Cache。

注意力、归一化、前馈网络和Decoder分别实现，可以逐个阅读和测试。实验主要对照缓存解码与完整前向计算，检查缓存的分配和更新方式，并测量prefill与decode的耗时。

## 运行训练示例

需要Python 3.10+和PyTorch 2.12.1。在虚拟环境中运行：

```bash
git clone https://github.com/SeRendizc/decoder-inference-lab.git
cd decoder-inference-lab
python -m pip install -e ".[dev]"
python scripts/train_tiny_corpus.py
```

示例使用两层模型，对一段重复的短文本训练200步，然后输出loss、token准确率和续写结果。它在CPU上运行，可以快速检查训练与生成流程。模型和训练参数都定义在[脚本内](scripts/train_tiny_corpus.py)。

使用CUDA时，需要自行安装对应平台的PyTorch wheel；项目不会自动选择CUDA包源。

## 模型

Decoder使用多头因果注意力、RMSNorm、Pre-norm残差块、MLP，以及可学习的token和位置嵌入。训练使用字节级UTF-8 tokenizer和next-token交叉熵。

```text
Token ID -> token嵌入 + 位置嵌入
         -> [RMSNorm -> attention -> 残差连接
             RMSNorm -> MLP       -> 残差连接] × 层数
         -> RMSNorm -> LM head -> logits
```

前向计算可以从[attention.py](src/decoder_inference_lab/model/attention.py)和[decoder.py](src/decoder_inference_lab/model/decoder.py)开始读。[generation.py](src/decoder_inference_lab/generation.py)负责贪心解码、EOS和上下文窗口处理。仓库提供模型实现与小规模实验，不包含预训练权重。

## KV Cache

两种缓存共用模型的前向计算逻辑：

- **动态缓存：**逐层保存Key/Value，解码时拼接新token对应的张量。
- **静态缓存：**预分配固定容量的张量，原位写入新增Key/Value，用`valid_length`记录有效区域。

生成时通过`cache_implementation="dynamic"`或`"static"`选择缓存。先对prompt做prefill，之后每次只输入一个新token；上下文窗口满后，再用最新窗口重建缓存。

静态缓存实现位于[cache.py](src/decoder_inference_lab/model/cache.py)，张量形状为`[batch, heads, capacity, head_dim]`。重置缓存会保留已分配的存储。

检查模型与缓存的配合：

```bash
python -m pytest tests/test_static_kv_cache_decoder.py -q
```

测试会对照缓存与完整前向的logits、静态与动态缓存的结果，并检查存储是否复用；同时覆盖层间长度不一致和容量不足的情况。[生成测试](tests/test_static_kv_cache_generation.py)则从完整解码循环检查这些行为。

## 实验

目前有三个对照脚本：

```bash
python scripts/profile_kv_cache.py
python scripts/profile_prefill_decode.py
python scripts/profile_cache_implementations.py
```

`profile_kv_cache.py`对照缓存生成与每轮重新计算上下文；`profile_prefill_decode.py`测量不同上下文长度下的prompt prefill、单token decode和KV张量大小；`profile_cache_implementations.py`先检查输出一致，再比较动态与静态缓存的生成耗时。

脚本包含预热、重复计时和CUDA同步。CUDA可用时优先使用GPU，否则使用CPU；设备性能有限时，可以调小脚本中的模型规模。动态/静态对照的耗时包含缓存分配，报告中的KV大小只统计缓存张量，不是整体显存占用。发布结果时，请保留对应的硬件、dtype和序列长度。

仓库中的[环境记录](artifacts/d1_environment.json)来自WSL2、RTX 3060 Laptop GPU和PyTorch 2.12.1+cu130。检查自己的环境：

```bash
python scripts/verify_environment.py --check-compile --output artifacts/local_environment.json
```

## 测试与后续实验

```bash
python -m pytest tests -q
python -m ruff check .
```

测试覆盖模型形状、梯度、因果性、训练和缓存行为。没有CUDA时，CUDA专项测试会跳过。

接下来计划做注意力kernel对照和profiler trace，再测量不同dtype与`torch.compile`的影响。现有脚本针对本地模型实验，尚不涉及并发推理服务的压测。

相关项目：[Agent Eval Lab](https://github.com/SeRendizc/agent-eval-lab)测试模型服务API，[Agent Runtime Lab](https://github.com/SeRendizc/agent-runtime-lab)研究工具执行与恢复机制。
