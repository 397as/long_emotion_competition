import os
import json
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.utils.data import Dataset

# Unsloth imports
from unsloth import FastLanguageModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    DataCollatorForLanguageModeling
)

EMOTION_LABELS_28 = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]


# -----------------------------
# 读取与清洗 GoEmotions CSV
# -----------------------------
def load_goemotions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "text" in df.columns, "CSV 中必须包含 'text' 列。"

    # 检查 28 标签列是否齐全
    miss = [c for c in EMOTION_LABELS_28 if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少标签列: {miss}")

    # 过滤 very_unclear 与 全零标签
    if "example_very_unclear" in df.columns:
        df = df[(df["example_very_unclear"] != True) & (df["example_very_unclear"] != "TRUE")]

    label_mat = df[EMOTION_LABELS_28].values
    keep = label_mat.sum(axis=1) > 0
    df = df[keep].reset_index(drop=True)

    # 去除空文本
    df = df[df["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    return df


# -----------------------------
# EC（多标签）Dataset & 指标
# -----------------------------
class ECDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df[EMOTION_LABELS_28].astype(int).values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)  # 多标签 -> float32
        return item


def ec_compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    exact_match = (preds == labels).all(axis=1).mean()
    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "exact_match_acc": exact_match}


# -----------------------------
# ED（N+1 段，找"异类"）构造
# -----------------------------
def _main_label_idx(row: pd.Series) -> Optional[int]:
    pos = [i for i, v in enumerate(row[EMOTION_LABELS_28].tolist()) if v == 1]
    if not pos:
        return None
    return random.choice(pos)  # 多标签时随机选一个主情感


def build_ed_packs(df: pd.DataFrame, K: int, max_examples: int, seed: int = 42) -> List[Tuple[str, int]]:
    random.seed(seed); np.random.seed(seed)
    work = df.copy()
    work["main_label"] = work.apply(_main_label_idx, axis=1)
    work = work[~work["main_label"].isna()].reset_index(drop=True)
    work["main_label"] = work["main_label"].astype(int)

    groups = {i: sub for i, sub in work.groupby("main_label")}
    all_labels = sorted(groups.keys())

    def sample_same(label_idx, n):
        pool = groups[label_idx]
        if len(pool) < n:
            return pool.sample(n, replace=True, random_state=seed)["text"].tolist()
        return pool.sample(n, replace=False, random_state=seed)["text"].tolist()

    def sample_diff(label_idx):
        others = [x for x in all_labels if x != label_idx]
        if not others:
            return None
        other = random.choice(others)
        return groups[other].sample(1, replace=True, random_state=seed)["text"].iloc[0]

    packs = []
    target = min(max_examples, len(work))
    for _ in range(target):
        label_idx = random.choice(all_labels)
        same_texts = sample_same(label_idx, K - 1)
        diff_text = sample_diff(label_idx)
        if diff_text is None:
            continue

        segs = same_texts + [diff_text]
        idxs = list(range(K))
        random.shuffle(idxs)
        shuffled = [segs[i] for i in idxs]
        odd_index = idxs.index(K - 1)  # 原最后一个即"异类"

        numbered = [f"Segment {i+1}: {t}" for i, t in enumerate(shuffled)]
        pack = "\n".join(numbered) + f"\nQuestion: Which segment expresses a different emotion? Answer with index 1..{K}."
        packs.append((pack, odd_index))
    return packs


class EDDataset(Dataset):
    def __init__(self, packs: List[Tuple[str, int]], tokenizer, max_len: int):
        self.packs = packs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, idx):
        text, label_idx = self.packs[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_idx, dtype=torch.long)  # 多类索引
        return item


def ed_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


# -----------------------------
# 训练入口
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="both", choices=["ec", "ed", "both"])
    parser.add_argument("--model_path", type=str, default="/data/zhangjingwei/LL-Doctor-qwen3-8b-Model")
    parser.add_argument("--data_csv", type=str, default="/data/zhangjingwei/data/goemotions.csv")
    parser.add_argument("--output_dir", type=str, default="/data/zhengjb/results_x")
    parser.add_argument("--max_len", type=int, default=64)     # GoEmotions <= 30，留冗余
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=128)   # Unsloth优化后可以使用更大的batch_size
    parser.add_argument("--seed", type=int, default=42)
    # ED 相关
    parser.add_argument("--ed_k", type=int, default=5, help="每个ED样本包含的段落数（N+1），默认5")
    parser.add_argument("--ed_max_train", type=int, default=50000)
    parser.add_argument("--ed_max_eval", type=int, default=5000)
    
    # Unsloth 相关参数
    parser.add_argument("--load_in_4bit", action="store_true", help="使用4-bit量化")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data: {args.data_csv}")
    df = load_goemotions_csv(args.data_csv)
    
    # 使用Unsloth加载模型和tokenizer
    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_len * 2,  # 为ED任务预留更多长度
        dtype=None,  # 自动选择最优精度
        load_in_4bit=args.load_in_4bit,  # 4-bit量化节省显存
    )

    # ------------------ EC ------------------
    if args.task in ("ec", "end", "both"):
        print("\n>>> Training EC (multi-label 28)")
        tmp = df.copy()
        tmp["main_label"] = tmp[EMOTION_LABELS_28].idxmax(axis=1)
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=args.seed, stratify=tmp["main_label"])

        ec_train = ECDataset(train_df, tokenizer, args.max_len)
        ec_val = ECDataset(val_df, tokenizer, args.max_len)

        out_dir = os.path.join(args.output_dir, "ec")
        os.makedirs(out_dir, exist_ok=True)

        # 配置LoRA适配器
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth优化的梯度检查点
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )

        # 配置训练参数 - Unsloth优化版本
        ec_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # 梯度累积
            learning_rate=2e-4,  # LoRA通常使用更高的学习率
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="steps",
            load_best_model_at_end=False,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),  # 自动选择精度
            bf16=torch.cuda.is_bf16_supported(),
            warmup_steps=100,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            save_steps=500
        )

        ec_trainer = Trainer(
            model=model,
            args=ec_args,
            train_dataset=ec_train,
            eval_dataset=ec_val,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            compute_metrics=ec_compute_metrics
        )


        # 使用Unsloth优化的训练
        print("Starting EC training with Unsloth...")
        ec_trainer.train()

        print("Saving model...")
        # 保存模型
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        metrics = ec_trainer.evaluate()
        print("EC eval:", metrics)
        

        # 保存验证集预测
        preds = ec_trainer.predict(ec_val)
        probs = 1 / (1 + np.exp(-preds.predictions))
        bin_pred = (probs >= 0.5).astype(int)
        pred_path = os.path.join(out_dir, "val_predictions.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for i in range(len(ec_val)):
                rec = {
                    "text": val_df.iloc[i]["text"],
                    "pred": {EMOTION_LABELS_28[j]: int(bin_pred[i, j]) for j in range(28)}
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"EC predictions saved -> {pred_path}")

    # ------------------ ED ------------------
    if args.task in ("ed", "both"):
        print("\n>>> Training ED (odd-one-out)")
        
        # 重新加载模型用于ED任务
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=max(128, args.max_len * 2),
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )
        
        tmp = df.copy()
        tmp["main_label"] = tmp[EMOTION_LABELS_28].idxmax(axis=1)
        base_train, base_val = train_test_split(df, test_size=0.1, random_state=args.seed, stratify=tmp["main_label"])

        train_packs = build_ed_packs(base_train, K=args.ed_k, max_examples=args.ed_max_train, seed=args.seed)
        val_packs = build_ed_packs(base_val, K=args.ed_k, max_examples=args.ed_max_eval, seed=args.seed + 1)

        # ED 输入较长，这里放宽长度
        ed_train = EDDataset(train_packs, tokenizer, max_len=max(128, args.max_len * 2))
        ed_val = EDDataset(val_packs, tokenizer, max_len=max(128, args.max_len * 2))

        out_dir = os.path.join(args.output_dir, "ed")
        os.makedirs(out_dir, exist_ok=True)

        # 配置LoRA适配器 - ED任务
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )

        ed_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,  # ED任务使用稍高的学习率
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="steps",
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            warmup_steps=100,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            save_steps=500
        )

        ed_trainer = Trainer(
            model=model,
            args=ed_args,
            train_dataset=ed_train,
            eval_dataset=ed_val,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            compute_metrics=ed_compute_metrics
        )


        print("Starting ED training with Unsloth...")
        ed_trainer.train()

        print("Saving model...")
        # 保存模型
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        metrics = ed_trainer.evaluate()
        print("ED eval:", metrics)
        
        

        # 保存验证集预测
        preds = ed_trainer.predict(ed_val)
        pred_idx = preds.predictions.argmax(axis=1).tolist()
        pred_path = os.path.join(out_dir, "val_predictions.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for i, (pack, gold) in enumerate(val_packs):
                f.write(json.dumps({
                    "pack_text": pack,
                    "gold_index": int(gold),
                    "pred_index": int(pred_idx[i])
                }, ensure_ascii=False) + "\n")
        print(f"ED predictions saved -> {pred_path}")


if __name__ == "__main__":
    main()
