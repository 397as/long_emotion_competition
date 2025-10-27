## CloudCom 2025 情感咨询对话挑战实现

该项目提供一个完整的多轮情绪咨询 RAG + LLM 生成流水线，读取 `data/Conversations_Long.jsonl`，构建向量检索，调用本地 vLLM 推理服务生成心理咨询师回复，并输出到 `outputs/Emotion_Conversatin_Result.jsonl`。

### 依赖安装（uv）

```bash
uv sync
```

### 运行

```bash
python -m src.runner_mc --config src/config.yaml \
    --data data/Conversations_Long.jsonl \
    --output outputs/Emotion_Conversatin_Result.jsonl
```

运行前请确保配置文件中的 LLM 推理端点（默认为 `http://127.0.0.1:8000/generate`）可用，并已部署兼容 OpenAI/vLLM 的接口。