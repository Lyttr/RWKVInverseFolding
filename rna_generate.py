#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import orjson
import pandas as pd
from tqdm import tqdm


# ------------------------------
# 工具函数
# ------------------------------

def generate_random_rna_sequence(n: int) -> str:
    return ''.join(random.choices('AUGC', k=n))


def mutate_sequence(seq: str, rate: float = 0.1) -> str:
    n = len(seq)
    k = max(1, int(n * rate))
    idxs = random.sample(range(n), k)
    bases = ('A', 'U', 'G', 'C')
    seq_list = list(seq)
    for i in idxs:
        orig = seq_list[i]
        choices = [b for b in bases if b != orig]
        seq_list[i] = random.choice(choices)
    return ''.join(seq_list)


def save_json(data, path: str):
    with open(path, "wb") as f:
        f.write(orjson.dumps(data))


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _extract_dot(token: str):
    return token if set(token) <= set(".()") else None


# ------------------------------
# RNAfold 调用（单次与分片）
# ------------------------------

def _fold_single_call(
    seqs,
    rnafold_cmd: str = "RNAfold",
    jobs: Optional[int] = None,
    timeout: Optional[int] = None,
    quiet: bool = False,
):
    """
    执行一次 RNAfold 调用（不做分片）
    - jobs>1 时使用 -j 并行且传入 OMP_NUM_THREADS
    - timeout 为每次子进程的超时（秒）
    """
    if not seqs:
        return []

    exe = shutil.which(rnafold_cmd)
    if exe is None:
        raise RuntimeError(f"[RNAfold] not found: {rnafold_cmd}. "
                           f"Use --rnafold_cmd=/abs/path/to/RNAfold or fix PATH.")

    use_threads = isinstance(jobs, int) and jobs > 1
    cmd = [exe, "--noPS"]
    if use_threads:
         cmd += [f"-j{jobs}"]

    # 传递 OpenMP 相关环境变量到子进程
    env = os.environ.copy()
    if use_threads:
        env["OMP_NUM_THREADS"] = str(jobs)
        env.setdefault("OMP_PROC_BIND", "spread")
        env.setdefault("OMP_PLACES", "cores")

    # 输入拼接
    inp = "\n".join(seqs) + "\n"

    if not quiet:
        print(f"[RNAfold] cmd: {' '.join(cmd)} | OMP_NUM_THREADS={env.get('OMP_NUM_THREADS', '')} | "
              f"batch={len(seqs)} | timeout={timeout or 'None'}",
              file=sys.stderr)

    try:
        res = subprocess.run(
            cmd,
            input=inp,
            text=True,
            capture_output=True,
            timeout=timeout,   # 每片的独立超时
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"[RNAfold] timeout after {timeout}s (batch size={len(seqs)})")

    # 若并行失败则回退串行一次
    if res.returncode != 0 and use_threads:
        if not quiet:
            print("[RNAfold] parallel failed, fallback to serial...", file=sys.stderr)
        res = subprocess.run(
            [exe, "--noPS"],
            input=inp,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
  
    if res.returncode != 0:
        err = (res.stderr or "").strip()
        raise RuntimeError(f"[RNAfold] failed (batch={len(seqs)}). STDERR:\n{err}")

    # 解析输出：每条序列对应两行（第一行为输入/标题，第二行以 dot-bracket 开头）
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    dots, i = [], 0
    while i < len(lines):
        if i + 1 < len(lines):
            tok = (lines[i + 1].split() or [""])[0]
            dots.append(_extract_dot(tok))
            i += 2
        else:
            dots.append(None)
            i += 1

    if len(dots) < len(seqs):
        dots += [None] * (len(seqs) - len(dots))
    elif len(dots) > len(seqs):
        dots = dots[:len(seqs)]

    return dots


def fold_batch(
    seqs,
    rnafold_cmd: str = "RNAfold",
    jobs: Optional[int] = None,
    timeout: Optional[int] = None,
    chunk: int = 256,
    quiet: bool = False,
):
    """
    对很大的 seqs 自动分片，每片调用一次 _fold_single_call
    - chunk: 每片最大条数（默认 256，可用 --fold_chunk 调整）
    """
    if not seqs:
        return []

    out = []
    for i in range(0, len(seqs), chunk):
        part = seqs[i:i + chunk]
        dots = _fold_single_call(
            part,
            rnafold_cmd=rnafold_cmd,
            jobs=jobs,
            timeout=timeout,
            quiet=quiet,
        )
        
        out.extend(dots)
    print("fold_batch finish")
    return out


# ------------------------------
# 数据集生成
# ------------------------------

def generate_train_set(
    test_structures,
    output_dir: str,
    train_size: int,
    chunk_size: int,
    num_threads: int,         # 供 UI/参数展示；真正生效的是 fold_jobs
    min_len: int,
    max_len: int,
    rnafold_cmd: str,
    similar_per_seq: int = 99,
    mutate_rate: float = 0.10,
    batch_bases: int = 256,
    fold_jobs: Optional[int] = None,      # >1 才并行；None/<=1 串行
    fold_timeout: Optional[int] = None,   # 每片超时（秒）
    fold_chunk: int = 256,                # 每次 RNAfold 最大条数
    quiet: bool = False,                  # 关闭每片 stderr 提示
):
    train_dir = os.path.join(output_dir, "train_chunks")
    os.makedirs(train_dir, exist_ok=True)

    existing = sorted([f for f in os.listdir(train_dir) if f.endswith(".json")])
    completed = len(existing) * chunk_size
    print(f"Completed: {completed}, Target: {train_size}")

    chunk_idx = len(existing)

    # 将 fold_jobs 规范化（只允许 >1 生效）
    eff_jobs = fold_jobs if (isinstance(fold_jobs, int) and fold_jobs > 1) else None
    if eff_jobs:
        print(f"[Info] RNAfold parallel jobs = {eff_jobs} (OMP_NUM_THREADS={eff_jobs})")
    else:
        print(f"[Info] RNAfold running in serial mode")

    def rand_base():
        return generate_random_rna_sequence(random.randint(min_len, max_len))

    try:
        with tqdm(total=train_size, desc="Generating training set", initial=completed) as pbar:
            while completed < train_size:
                train_data = []

                while len(train_data) < chunk_size and completed < train_size:
                    # 1) 采样候选序列（基础 + 变异）
                    base_seqs = [rand_base() for _ in range(batch_bases)]
                    cand_seqs = []
                    cand_seqs.extend(base_seqs)
                    for s in base_seqs:
                        for _ in range(similar_per_seq):
                            cand_seqs.append(mutate_sequence(s, mutate_rate))

                    # 2) 调用 RNAfold（分片 + 并行 + 超时）
                    cand_dots = fold_batch(
                        cand_seqs,
                        rnafold_cmd=rnafold_cmd,
                        jobs=eff_jobs,
                        timeout=fold_timeout,
                        chunk=fold_chunk,
                        quiet=quiet,
                    )
                   
                    # 3) 过滤 & 装入
                    grew = 0
                    for seq, dot in zip(cand_seqs, cand_dots):
                       
                        if dot is None:
                        
                            continue
                        if dot in test_structures:
                            continue
                        train_data.append({"sequence": seq, "structure": dot})
                        completed += 1
                        grew += 1
                        pbar.update(1)
                        if len(train_data) >= chunk_size or completed >= train_size:
                            break
                    if grew == 0:
                        print("[warn] this batch produced 0 usable samples; consider tuning max_len/batch size.")

                # 4) 落盘一个 chunk
                if train_data:
                    if len(train_data) > chunk_size:
                        train_data = train_data[:chunk_size]
                    path = os.path.join(train_dir, f"train_chunk_{chunk_idx:05d}.json")
                    save_json(train_data, path)
                    print(f"Saved chunk to {path} (size={len(train_data)})")
                    chunk_idx += 1

    except KeyboardInterrupt:
        # 中断时把当前部分也保存掉
        if 'train_data' in locals() and train_data:
            path = os.path.join(train_dir, f"train_chunk_{chunk_idx:05d}.json")
            save_json(train_data, path)
            print(f"\n[Interrupted] Partial chunk saved to {path} (size={len(train_data)})")
        raise

    # 合并
    print(f"Training set generated target: {train_size}, saved chunks under {train_dir}/")
    merged = []
    for fname in sorted(os.listdir(train_dir)):
        if fname.endswith(".json"):
            data = load_json(os.path.join(train_dir, fname))
            merged.extend(data)
    merged_train_path = os.path.join(output_dir, "rna_train.json")
    save_json(merged, merged_train_path)
    print(f"Chunks merged to {merged_train_path}, total: {len(merged)} samples")


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate RNA dataset using RNAfold with similar-sequence augmentation (batched & parallel)"
    )
    parser.add_argument("--output_dir", type=str, default="/pvc/dataset/8M0901")
    parser.add_argument("--testraw_dir", type=str, default="eterna100_puzzles.tsv")
    parser.add_argument("--train_size", type=int, default=8_000_000)
    parser.add_argument("--chunk_size", type=int, default=10_000)
    parser.add_argument("--min_len", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=400)

    # 并行参数
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--fold_jobs", type=int, default=None, help="override number of -j threads; >1 to enable")
    parser.add_argument("--rnafold_cmd", type=str, default="RNAfold")

    # 增广/批处理参数
    parser.add_argument("--similar_per_seq", type=int, default=99)
    parser.add_argument("--mutate_rate", type=float, default=0.10)
    parser.add_argument("--batch_bases", type=int, default=8)

    # 调用控制
    parser.add_argument("--fold_timeout", type=int, default=120, help="per-chunk timeout seconds")
    parser.add_argument("--fold_chunk", type=int, default=200, help="max sequences per RNAfold call")
    parser.add_argument("--quiet", action="store_true", help="suppress per-call stderr info")

    args = parser.parse_args()

    # 计算有效的 -j 线程数（>1 才生效）
    eff_jobs: Optional[int] = None
    if args.fold_jobs and args.fold_jobs > 1:
        eff_jobs = args.fold_jobs
    elif args.num_threads and args.num_threads > 1:
        eff_jobs = args.num_threads

    # 读取测试集结构并保存
    test_v2_path_compat = os.path.join(args.output_dir, "rna_test.json")
    test_v1_path = os.path.join(args.output_dir, "rna_test_v1.json")

    df = pd.read_csv(args.testraw_dir, sep="\t")
    v2_structures = set()
    if "Secondary Structure V2" in df.columns:
        for structure in df["Secondary Structure V2"].dropna():
            if len(structure) <= args.max_len:
                v2_structures.add(structure)
    v1_structures = set()
    if "Secondary Structure V1" in df.columns:
        for structure in df["Secondary Structure V1"].dropna():
            if len(structure) <= args.max_len:
                v1_structures.add(structure)

    os.makedirs(args.output_dir, exist_ok=True)
    save_json([{"structure": s} for s in v2_structures], test_v2_path_compat)
    print(f"Test set (V2) saved: {len(v2_structures)} structures → {test_v2_path_compat}")
    save_json([{"structure": s} for s in v1_structures], test_v1_path)
    print(f"Test set (V1) saved: {len(v1_structures)} structures → {test_v1_path}")

    union_structures = v1_structures | v2_structures
    print(f"Union test structures: {len(union_structures)} (V1∪V2)")

    generate_train_set(
        test_structures=union_structures,
        output_dir=args.output_dir,
        train_size=args.train_size,
        chunk_size=args.chunk_size,
        num_threads=args.num_threads,
        min_len=args.min_len,
        max_len=args.max_len,
        rnafold_cmd=args.rnafold_cmd,
        similar_per_seq=args.similar_per_seq,
        mutate_rate=args.mutate_rate,
        batch_bases=args.batch_bases,
        fold_jobs=eff_jobs,                # <- 真正生效的 -j
        fold_timeout=args.fold_timeout,
        fold_chunk=args.fold_chunk,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
