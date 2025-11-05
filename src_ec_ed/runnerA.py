#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, math, threading, queue, subprocess, signal, re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, MofNCompleteColumn, SpinnerColumn
from rich.live import Live
from rich.panel import Panel

console = Console()

def human_int(x):
    return f"{x:,}"

def get_gpu_mem():
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"], timeout=1.5)
        vals = out.decode().strip().splitlines()[0].split(",")
        used, total = [int(v.strip()) for v in vals[:2]]
        return used, total
    except Exception:
        return None, None

def torch_mem():
    if torch.cuda.is_available():
        used = int(torch.cuda.memory_allocated()/1024/1024)
        reserved = int(torch.cuda.memory_reserved()/1024/1024)
        return used, reserved
    return None, None

# ---------- Task-specific prompt builders (robust to common field names) ----------
def _get(d: Dict[str, Any], cand: List[str], default=""):
    for k in cand:
        if k in d: return d[k]
    return d.get("input", default)
def extract_segments(sample: Dict[str, Any]) -> List[str]:
    """
    万能段落提取（优先适配 ED 的 text=[{index, context}, ...] 结构）：
    1) 先检查常见键（包含 'text'）
    2) 列表元素是 dict 时，优先抓 context/ctx/utterance/text/sentence/content/value
       并按 index 排序（如果存在）
    3) 递归扫描整个样本，尝试找到字符串列表；找不到再将大字符串按常见分隔符切段
    """

    # 0) 一个小工具：从 list[dict/str] 抽出段，并按 index 排序（如有）
    def _extract_from_list(lst):
        idx_txt = []
        plain = []
        for it in lst:
            if isinstance(it, str) and it.strip():
                plain.append(it.strip())
            elif isinstance(it, dict):
                txt = (it.get("context") or it.get("ctx") or it.get("utterance") or
                       it.get("text") or it.get("sentence") or it.get("content") or it.get("value"))
                if isinstance(txt, str) and txt.strip():
                    idx_txt.append((it.get("index", 0), txt.strip()))
        if idx_txt:
            idx_txt.sort(key=lambda x: x[0])
            return [t for _, t in idx_txt]
        if plain:
            return plain
        return []

    # 1) 常见顶层键（加入了 'text'）
    prefer_keys = ["text","segments","candidates","options","sentences","spans","items","choices",
                   "texts","paragraphs","list","choices_text","context_list","ctx_list"]
    for k in prefer_keys:
        if k in sample:
            v = sample[k]
            if isinstance(v, list):
                segs = _extract_from_list(v)
                if len(segs) >= 2:
                    return segs

    # 2) 递归收集候选（字符串列表 / 字典列表）
    buckets = []

    def walk(x):
        if isinstance(x, list):
            segs = _extract_from_list(x)
            if segs:
                buckets.append(segs)
            for it in x:
                walk(it)
        elif isinstance(x, dict):
            # 先看子键里是否有 options/choices/... 这种典型容器
            inner = (x.get("options") or x.get("choices") or x.get("segments") or
                     x.get("sentences") or x.get("text") or x.get("paragraphs") or x.get("list"))
            if isinstance(inner, list):
                segs = _extract_from_list(inner)
                if segs:
                    buckets.append(segs)
            for _, v in x.items():
                walk(v)
        elif isinstance(x, str):
            s = x.strip()
            if s:
                buckets.append([s])

    walk(sample)

    # 2.1 先返回本就 >=2 的字符串列表
    for lst in buckets:
        if len(lst) >= 2:
            return lst

    # 2.2 把“单个大字符串”按分隔符切段
    def _smart_split(big: str) -> list:
        seps = ["\n\n", "|||", "<sep>", "<SEP>", "\n-\n", "\n—\n", "\n*\n"]
        for sp in seps:
            if sp in big:
                parts = [p.strip() for p in big.split(sp)]
                parts = [p for p in parts if p]
                if len(parts) >= 2:
                    return parts
        lines = [ln.strip() for ln in big.splitlines() if ln.strip()]
        # 形如 [0]/0)/1./2: 的编号行
        group, cur = [], []
        for ln in lines:
            if re.match(r"^\s*(\[\d+\]|\d+[\).:])\s+", ln):
                if cur: group.append(" ".join(cur).strip()); cur = []
                cur.append(ln)
            else:
                cur.append(ln)
        if cur: group.append(" ".join(cur).strip())
        if len(group) >= 2:
            clean = [re.sub(r"^\s*(\[\d+\]|\d+[\).:])\s+", "", g).strip() for g in group]
            clean = [c for c in clean if c]
            if len(clean) >= 2:
                return clean
        if len(lines) >= 2:
            return lines
        return [] 

    for lst in buckets:
        if len(lst) == 1:
            parts = _smart_split(lst[0])
            if len(parts) >= 2:
                return parts

    # 3) 兜底：返回空，交给外层构造占位
    return []


def build_prompt(task: str, sample: Dict[str, Any]) -> str:
    # 尽量兼容 LongEmotion 的统一样式；若字段名不同，可在这里微调
    if task == "emotion_classification":
        ctx = _get(sample, ["context","text","document","passage"])
        options = [
            ("A", "Anger"),
            ("B", "Anxiety"),
            ("C", "Depression"),
            ("D", "Frustration"),
            ("E", "Jealousy"),
            ("F", "Guilt"),
            ("G", "Fear"),
            ("H", "Embarrassment"),
        ]
        opt_str = "\n".join([f"{a}) {b}" for a,b in options])
        return (
            "You are an expert in long-context emotion understanding.\n"
            "Task: Read the long context and select the SINGLE best emotion for the TARGET (if target is absent, infer the dominant emotion in context).\n"
            "Choose ONE option and reply with the LETTER ONLY (A-H). Do NOT add any other text.\n\n"
            f"Options:\n{opt_str}\n\n"
            f"Context:\n{str(ctx)}\n\n"
            "Answer (LETTER ONLY):"
        )
    elif task == "emotion_detection":
        segs = extract_segments(sample)
        n = len(segs)
        if n < 2:
            # 记录异常，仍然构造一个占位，避免崩
            segs = segs if segs else ["(missing segment)"]
            n = len(segs)

        lines = [f"[{i}] {segs[i]}" for i in range(n)]
        seg_txt = "\n".join(lines)

        # 明确要求：只回 0..n-1 的数字索引
        return (
            "You are given multiple text segments. Exactly ONE segment expresses a different emotion from the others.\n"
            f"TASK: Identify the odd-one-out by its INDEX. Return ONLY a number in 0..{n-1}.\n"
            "Do NOT add any explanation or text.\n\n"
            f"{seg_txt}\n\n"
            "Answer (INDEX ONLY):"
        )
    elif task == "emotion_qa":
        ctx = _get(sample, ["context","text","document","passage"])
        q = _get(sample, ["question","query","ask"])
        return (
            "Answer the question grounded ONLY in the provided context.\n"
            "Be concise.\n"
            f"Question: {q}\n"
            f"Context:\n{ctx}\n"
            "Answer:"
        )
    elif task == "emotion_summary":
        report = _get(sample, ["report","context","text","document"])
        return (
            "Summarize the case into five parts: (i) Causes, (ii) Symptoms, (iii) Treatment Process, "
            "(iv) Illness Characteristics, (v) Treatment Effect. Use bullet points.\n"
            f"Case:\n{report}\nSummary:"
        )
    elif task == "emotion_conversation":
        history = _get(sample, ["dialog","conversation","history","messages"])
        if isinstance(history, list):
            lines = []
            for turn in history:
                role = turn.get("role","user")
                content = turn.get("content", "")
                lines.append(f"{role.upper()}: {content}")
            dialog_txt = "\n".join(lines[-20:])
        else:
            dialog_txt = str(history)
        return (
            "You are a professional counseling assistant. Provide an empathetic, concise next reply.\n"
            "Avoid giving medical diagnosis. Use supportive tone.\n\n"
            f"{dialog_txt}\n\nASSISTANT:"
        )
    else:
        # fallback: echo the text field
        return str(_get(sample, ["context","text","document","passage","report","dialog","conversation"], ""))

# ---------- IO ----------
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)

def count_lines(path: str) -> int:
    try:
        import mmap
        with open(path, "r+", encoding="utf-8") as f:
            buf = mmap.mmap(f.fileno(), 0)
            return sum(1 for _ in iter(buf.readline, b""))  # type: ignore
    except Exception:
        return sum(1 for _ in open(path, "r", encoding="utf-8"))

def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

# ---------- Generation ----------
def run(args):
    console.rule("[bold]LongEmotion Runner")
    console.print(f"[bold green]Model[/]: {args.model_path}")
    console.print(f"[bold green]Data [/]: {args.test_path}")
    console.print(f"[bold green]Out  [/]: {args.out_path}")
    console.print(f"[bold green]Task [/]: {args.task_name}\n")

    Path(os.path.dirname(args.out_path)).mkdir(parents=True, exist_ok=True)
    err_log = args.out_path + ".errors.log"
    if os.path.exists(err_log): os.remove(err_log)

    # model / tokenizer
    load_t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    load_t1 = time.time()

    total = count_lines(args.test_path)
    console.print(f"[cyan]Loaded model in {load_t1-load_t0:.2f}s. Samples: {total}[/]\n")

    # gen_kwargs = dict(
    #     max_new_tokens=args.max_new_tokens,
    #     do_sample=True,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     repetition_penalty=1.05
    # )

    gen_kwargs = dict(
        max_new_tokens=1,      # 只需要输出 1 个 token（A~L）
        do_sample=False,       # ← 关闭采样，走贪婪解码
        temperature=0.0,       # ← 没用了，但设 0 保守
        top_p=1.0,             # ← 不影响贪婪
        repetition_penalty=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    task_gen = dict(gen_kwargs)
    if args.task_name == "emotion_detection":
        task_gen.update(
            max_new_tokens=2,   # 索引最多两位数
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )

    # progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("⏳"),
        TimeRemainingColumn(),
        expand=True
    )

    task_main = progress.add_task("[bold white]Running", total=total)
    toks_in_total = 0
    toks_out_total = 0
    t_start = time.time()

    def gpu_panel():
        u1, r1 = torch_mem()
        u2, t2 = get_gpu_mem()
        tbl = Table.grid(padding=1)
        tbl.add_column("Metric", justify="left")
        tbl.add_column("Value", justify="right")
        tbl.add_row("torch.cuda allocated (MB)", str(u1) if u1 is not None else "-")
        tbl.add_row("torch.cuda reserved  (MB)", str(r1) if r1 is not None else "-")
        if u2 is not None:
            tbl.add_row("nvidia-smi used/total (MB)", f"{u2} / {t2}")
        return Panel(tbl, title="GPU / Memory", border_style="blue")

    with Live(refresh_per_second=4) as live, progress:
        live.update(gpu_panel())
        for idx, sample in enumerate(iter_jsonl(args.test_path), start=1):
            t0 = time.time()
            prompt = build_prompt(args.task_name, sample)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_len)
            toks_in = int(enc["input_ids"].shape[-1])
            toks_in_total += toks_in
            enc = {k: v.to(model.device) for k, v in enc.items()}

            try:
                with torch.inference_mode():
                    out = model.generate(**enc, **task_gen)
                gen_ids = out[0][enc["input_ids"].shape[-1]:]
                toks_out = int(gen_ids.shape[-1])
                toks_out_total += toks_out
                text = tok.decode(out[0], skip_special_tokens=True)
                gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()

                # —— 任务级后处理 —— 
                LETTER2LABEL = {
                    "A": "Anger", "B": "Anxiety", "C": "Depression", "D": "Frustration",
                    "E": "Jealousy", "F": "Guilt", "G": "Fear", "H": "Embarrassment",
                }

                if args.task_name == "emotion_classification":
                    m = re.search(r"[A-Ha-h]", gen_text)
                    if m:
                        gen_text = LETTER2LABEL.get(m.group(0).upper(), "Unknown") + "," + gen_text
                    else:
                        gen_text = "Unknown"+ "," + gen_text


                elif args.task_name == "emotion_detection":
                    # ED：数字/字母 → 索引；裁剪到合法范围
                    segs_for_len = extract_segments(sample)
                    n = max(len(segs_for_len), 1)


                    m_num = re.search(r"\d+", gen_text)
                    if m_num:
                        idx = int(m_num.group(0))
                    else:
                        m_ch = re.search(r"[A-Za-z]", gen_text)
                        if m_ch:
                            idx = ord(m_ch.group(0).upper()) - ord('A')  # A->0, B->1, ...
                        else:
                            idx = 0

                    if idx < 0: idx = 0
                    if idx >= n: idx = n - 1
                    gen_text = str(idx)
                    

            except Exception as e:
                with open(err_log, "a", encoding="utf-8") as ef:
                    ef.write(f"{idx}\t{repr(e)}\n")
                gen_text = f"[ERROR] {repr(e)}"

            # write out (incremental)
            result = {
                "id": sample.get("id", str(idx)),
                "task": args.task_name,
                "prompt_tokens": toks_in,
                "output_tokens": toks_out if 'toks_out' in locals() else 0,
                "prompt_preview": prompt[:2000],
                "output": gen_text
            }
            append_jsonl(args.out_path, result)

            t1 = time.time()
            dt = t1 - t0
            # update progress
            progress.update(task_main, advance=1, description=f"[white]Running • sample {idx}/{total} • {dt:.2f}s")
            # live panel (throughput)
            elapsed = max(time.time() - t_start, 1e-6)
            tok_s = toks_out_total / elapsed

            # ---- 安全 ETA 计算，避免除零/负数 ----
            done = max(idx, 1)                 # 至少为 1，避免除零
            remaining = max(total - idx, 0)    # 不小于 0
            eta_sec = (elapsed / done) * remaining

            meta = Table.grid(padding=1)
            meta.add_column("Stat", justify="left")
            meta.add_column("Value", justify="right")
            meta.add_row("Avg input toks", f"{toks_in_total//done}")
            meta.add_row("Avg output toks", f"{toks_out_total//done}")
            meta.add_row("Throughput (tok/s)", f"{tok_s:.1f}")
            meta.add_row("ETA (approx)", f"{eta_sec:.1f}s")
            live.update(Panel(meta, title="Run Stats", border_style="green"))

            # update GPU panel occasionally
            if idx % 5 == 0:
                live.update(gpu_panel())

    console.rule("[bold]Done")
    console.print(f"Results: [bold]{args.out_path}[/]")
    if os.path.exists(err_log):
        console.print(f"[yellow]Errors logged at[/] {err_log}")

def parse_args():
    p = argparse.ArgumentParser(description="LongEmotion Runner with Rich Progress")
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--task_name", required=True,
                   choices=["emotion_classification","emotion_detection","emotion_qa","emotion_summary","emotion_conversation"])
    p.add_argument("--max_len", type=int, default=8192)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)

