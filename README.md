# Long Emotion MC & ES 流水线

本仓库实现了 CloudCom 2025 长文本情感咨询挑战的两条检索增强生成（RAG）流水线：

- **MC（Mental Counseling）**：面向多轮咨询对话的回应生成，支持自我评估改写。
- **ES（Emotion Summary）**：针对咨询案例材料抽取结构化情绪摘要。

两条流水线共享相同的依赖管理、向量检索组件与 LLM 调用辅助工具，可根据需求分别或组合使用。

## 技术栈概览
- **Python 3.12 + uv**：使用 uv 管理虚拟环境与依赖锁定，确保 MC / ES 流水线共享一致的运行时环境。
- **FAISS**：负责长文本分块后的向量检索，分别在 `indexes/mc/` 与 `indexes/es/` 中缓存索引结果以复用推理。
- **Sentence Transformers**：默认采用 `sentence-transformers` 模型抽取嵌入，用于召回与排序候选上下文。
- **vLLM / OpenAI Chat Completions API**：通过统一的 Chat Completions 接口调用 LLM，可对接自建 vLLM 推理服务或兼容的云端模型。
- **FlashInfer**：结合 `flashinfer-cubin` 与 `flashinfer-jit-cache` 以加速部分算子，提升在 vLLM 上的推理吞吐。

## 仓库结构
- `src_mc/`：情感咨询（MC）流水线的核心实现（检索器、提示词模板、LLM 编排与 CLI 入口）。详细说明见 [`src_mc/README.md`](src_mc/README.md)。
- `src_es/`：情绪摘要（ES）流水线的实现，包含结构化字段提示词与运行脚本。更多细节见 [`src_es/README.md`](src_es/README.md)。
- `data/`：输入的 JSONL 数据样例，如 `Conversations_Long.jsonl`（MC）与 `Emotion_Summary.jsonl`（ES）。
- `outputs/`：运行流水线后生成的结果文件，默认命名分别为 `Emotion_Conversation_Result.jsonl` 与 `Emotion_Summary_Result.jsonl`。
- `main.py`：用于实验或集成的便捷入口脚本。
- `pyproject.toml` / `uv.lock`：使用 [uv](https://github.com/astral-sh/uv) 管理的 Python 依赖定义。

## 环境准备
项目基于 **Python 3.12**，并使用 [uv](https://github.com/astral-sh/uv) 管理虚拟环境与依赖。首次使用时建议按以下步骤完成准备：

1. **安装系统依赖**：确保已安装 Git 与 Python 3.12（Linux 用户可通过 `apt`/`yum`，macOS 用户可通过 Homebrew）。推荐额外安装 `build-essential` / `xcode-select --install` 以编译少量含 C 扩展的依赖。
2. **安装 uv**：若尚未安装，可执行 `pip install uv` 或参考官方文档获取预编译发行版。
3. **同步项目依赖**：在仓库根目录执行：

   ```bash
   uv sync
   ```

   该命令会创建隔离的 `.venv` 环境，并根据 `pyproject.toml` 与 `uv.lock` 解析全部依赖。
4. **激活虚拟环境（可选）**：如需直接使用解释器，可运行 `source .venv/bin/activate`（Windows 为 `.venv\Scripts\activate`）。

后续运行示例、单元测试或脚本时，均可在命令前添加 `uv run` 前缀，以保证使用到正确的虚拟环境与锁定依赖。

## 准备推理端点
流水线假设存在一个兼容 OpenAI Chat Completions 的接口。若本地部署 [vLLM](https://github.com/vllm-project/vllm)，可参考如下命令：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /data/zhangjingwei/LL-Doctor-qwen3-8b-Model \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768
```

根据实际硬件调整显存相关参数；若服务地址不同，请同步更新配置文件中的 `llm.endpoint`。

## 运行情感咨询（MC）流水线
使用默认配置处理示例数据：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m src_mc.runner_mc \
  --config src_mc/config.yaml \
  --data data/Conversations_Long.jsonl \
  --output outputs/Emotion_Conversation_Result.jsonl
```

首次运行会在 `indexes/mc/` 目录下生成 FAISS 索引及文本缓存，如需重建可删除后重新执行。启用 `self_judge` 配置块后，流水线会执行一次基于督导提示词的自我评估，并在得分低于阈值时进行改写。

## 运行情绪摘要（ES）流水线
ES 流水线会将案例材料拼接后检索相关片段，并生成五个结构化字段。使用默认配置处理示例数据：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m src_es.runner_es \
  --config src_es/config_es.yaml \
  --data data/Emotion_Summary.jsonl \
  --output outputs/Emotion_Summary_Result.jsonl
```

如启用向量检索，首次运行会在 `indexes/es/` 下构建索引。可通过 `per_field_queries` 为各字段自定义检索关键词，或在配置中将 `use_rag` 置为 `false` 以仅基于原始案例文本生成。

## 配置说明
- `src_mc/config.yaml` 与 `src_es/config_es.yaml` 分别控制 MC、ES 两条流水线的检索参数、嵌入模型及 LLM 生成参数。
- JSONL 读写、对话抽取等共享工具函数位于 `src_mc/utils.py`；向量检索逻辑实现于 `src_mc/retriever.py`；与推理服务交互的辅助函数位于 `src_mc/llm.py`。

如需更详尽的流程说明，请查阅对应子目录下的 README。
