# Repository Guidelines

## 项目结构与模块组织

本仓库依赖 Python 3.12，并通过 `uv`（`pyproject.toml`, `uv.lock`）管理环境。`src_mc/` 实现咨询对话 RAG 流水线，`src_es/` 负责情绪摘要，两者复用 `src_mc/utils.py` 与  
 `src_mc/retriever.py`。示例输入位于 `data/`，运行结果写入 `outputs/`，FAISS 索引与嵌入缓存保存在 `indexes/`（需保持在本地，勿提交）。临时脚本可置于 `scripts/`，以免污染
核心模块。

## 构建、测试与开发命令

首次执行 `uv sync` 安装依赖。MC 流水线示例：`uv run python -m src_mc.runner_mc --config src_mc/config.yaml --data data/Conversations_Long.jsonl --output outputs/       
  Emotion_Conversatin_Result.jsonl`；ES 流水线改为 `src_es.runner_es`。调参时可追加 `--concurrency 4` 等 CLI 选项。若更新检索或提示配置，请删除对应 `indexes/mc/` 或  
 `indexes/es/` 后再运行，确保索引重建。

## 代码风格与命名约定

遵循 PEP 8，使用四空格缩进与类型注解，变量/函数采用 snake_case，类使用 PascalCase，提示常量保持大写（如 `MC_SYSTEM`）。避免 `print`，统一使用模块级 `LOGGER`。与用户交互
的文档或 docstring 建议保持中英双语；行内注释仅用于复杂逻辑。YAML 键名以小写无连字符为准，新增 CLI 参数命名与现有风格一致。

# 与用户交流风格
永远用中文回复，专业词汇保留英文