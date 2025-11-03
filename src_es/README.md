# Emotion Summary（ES）流水线说明

ES 流水线用于从咨询案例材料中抽取结构化的情绪信息，包括成因、症状、干预过程、病程特征与疗效等五个字段。该流水线复用 `src_mc` 中的检索与模型调用组件，可独立于 MC 流水线运行。

## 目录结构
- `runner_es.py`：命令行入口，负责读取案例、构建检索器、调用大模型并输出 JSONL 结果。
- `config_es.yaml`：默认配置文件，包含检索参数、嵌入模型、LLM 端点与各字段的检索查询词。
- `prompts_es.py`：系统提示词与用户提示词模板，约束模型按照指定字段输出 JSON。

## 运行步骤
1. **准备环境与推理端点**：按照仓库根目录 README 中的说明完成依赖安装，并确保 `config_es.yaml` 中的 `llm.endpoint` 可用。
2. **准备输入数据**：数据需为 JSONL 格式，单条记录至少包含 `id` 以及 `case_description`、`consultation_process`、`experience_and_reflection` 等字段。
3. **执行流水线**：

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python -m src_es.runner_es \
     --config src_es/config_es.yaml \
     --data data/Emotion_Summary.jsonl \
     --output outputs/Emotion_Summary_Result.jsonl
   ```

   首次运行会依据配置在 `indexes/es/` 下构建向量索引和文本缓存。

## 配置要点
- `es.use_rag`：是否启用基于 FAISS 的检索增强。关闭后模型将直接处理整合后的案例文本。
- `es.chunk_tokens` / `es.overlap_tokens`：构建索引时的切分策略。
- `per_field_queries`：为每个输出字段指定检索查询词，帮助检索器挑选最相关的片段。
- `llm`：包含模型名称、推理端点与生成参数，可根据实际部署环境调整。

## 输出格式
流水线会为每条输入案例输出一行 JSON，示例：

```json
{
  "id": "case_001",
  "predicted_cause": "长期加班导致情绪透支",
  "predicted_symptoms": "持续焦虑与睡眠障碍",
  "predicted_treatment_process": "使用认知重构与正念放松练习",
  "predicted_illness_Characteristics": "症状在项目高峰期加重，缺乏家庭支持",
  "predicted_treatment_effect": "四周后情绪稳定，睡眠质量提升"
}
```

若某字段缺乏明确证据，流水线会自动填充“未见明确描述”。
