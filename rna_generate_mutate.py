#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import subprocess
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from numpy.lib.format import open_memmap
import orjson

VOCAB = {
    ".": 0, "(": 1, ")": 2,
    "A": 3, "C": 4, "G": 5, "U": 6,
    "\n": 7, "PAD": 8
}

ID_PAD = VOCAB["PAD"]

def tokenize_item(item, ctx_len=256):
    text = item["structure"] + "\n" + item["sequence"]
    token_ids = [VOCAB[c] for c in text]
    if len(token_ids) < ctx_len:
        token_ids += [VOCAB["PAD"]] * (ctx_len - len(token_ids))
    else:
        token_ids = token_ids[:ctx_len]
    return np.array(token_ids, dtype=np.uint16)

def save_chunk_npy(train_data, path, ctx_len=256):

    arr = np.zeros((len(train_data), ctx_len), dtype=np.uint16)
    for i, item in enumerate(train_data):
        arr[i] = tokenize_item(item, ctx_len)
    np.save(path, arr)
    print(f"[chunk] Saved chunk to {path}.npy, shape={arr.shape}")

def merge_chunks_npy(train_dir, output_path, ctx_len=256, remove_chunks=False):
    files = sorted(
        [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".npy")]
    )
    if not files:
        raise FileNotFoundError(f"No .npy files found in {train_dir}")
    total_rows = 0
    ref_dtype = None
    for fp in files:
        arr = np.load(fp, mmap_mode='r')
        if arr.ndim != 2:
            raise ValueError(f"{fp}: expected 2D array, got shape={arr.shape}")
        if arr.shape[1] != ctx_len:
            raise ValueError(f"{fp}: ctx_len mismatch, got {arr.shape[1]}, expect {ctx_len}")
        if ref_dtype is None:
            ref_dtype = arr.dtype
        elif arr.dtype != ref_dtype:
            raise ValueError(f"{fp}: dtype mismatch {arr.dtype} vs {ref_dtype}")
        total_rows += arr.shape[0]

    if total_rows == 0:
        raise ValueError("No rows to merge (total_rows == 0).")

    tmp_out = output_path + ".tmp"
    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    print(f"[merge] {len(files)} chunks, total rows={total_rows}, ctx_len={ctx_len}, dtype={ref_dtype}")
    print(f"[merge] writing to {tmp_out} (atomic rename to {output_path})")

    merged = open_memmap(
        tmp_out, mode='w+', dtype=ref_dtype, shape=(total_rows, ctx_len)
    )

    offset = 0
    for i, fp in enumerate(files):
        arr = np.load(fp, mmap_mode='r')
        n = arr.shape[0]
        merged[offset:offset+n] = arr
        offset += n
        if (i % 50) == 0 or i == len(files) - 1:
            print(f"[merge] copied {i+1}/{len(files)} chunks, rows={offset}/{total_rows}")
    del merged
    os.replace(tmp_out, output_path)
    print(f"[merge] done → {output_path} (shape=({total_rows}, {ctx_len}), dtype={ref_dtype})")

    if remove_chunks:
        for fp in files:
            try:
                os.remove(fp)
            except Exception as e:
                print(f"[warn] failed to remove {fp}: {e}")

def run_rnafold(seq, rnafold_cmd, timeout=3):

    try:
        result = subprocess.run(
            [rnafold_cmd, "--noPS"],
            input=seq + "\n",
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            dot = lines[1].split()[0]
            return dot
    except Exception:
        return None
    return None

def generate_random_rna_sequence(n):
    return ''.join(random.choices('AUGC', k=n))

def mutate_sequence(seq, rate=0.1):

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

def is_valid_structure(dot):

    return bool(dot and set(dot) != {'.'})

def generate_and_validate(test_structures, min_len, max_len, rnafold_cmd, similar_per_seq, mutate_rate):
    seq = generate_random_rna_sequence(random.randint(min_len, max_len))
    seqs = [seq] + [mutate_sequence(seq, mutate_rate) for _ in range(similar_per_seq)]
    results = []
    for s in seqs:
        dot = run_rnafold(s, rnafold_cmd)
        if is_valid_structure(dot) and dot not in test_structures:
            results.append({"sequence": s, "structure": dot})
    return results

def generate_train_set(test_structures, output_dir, train_size, chunk_size,
                       num_threads, min_len, max_len, rnafold_cmd,
                       similar_per_seq, mutate_rate, ctx_len):
    train_dir = os.path.join(output_dir, "train_chunks")
    os.makedirs(train_dir, exist_ok=True)


    existing_chunks = sorted([f for f in os.listdir(train_dir) if f.endswith(".npy")])
    completed = 0
    for f in existing_chunks:
        try:
            arr = np.load(os.path.join(train_dir, f), mmap_mode='r')
            if arr.ndim == 2:
                completed += arr.shape[0]
        except Exception:
            pass

    print(f"[gen] Completed: {completed}, Target: {train_size}")
    chunk_idx = len(existing_chunks)

    with tqdm(total=train_size, desc="Generating training set", initial=completed) as pbar:
        while completed < train_size:
            train_data = []
          
            n_tasks = int(chunk_size * 1.1)
            executor = ThreadPoolExecutor(max_workers=num_threads)
            futures = [
                executor.submit(
                    generate_and_validate,
                    test_structures, min_len, max_len,
                    rnafold_cmd, similar_per_seq, mutate_rate
                )
                for _ in range(n_tasks)
            ]

            try:
                for future in as_completed(futures):
                    items = future.result()
                    if items:
                        for it in items:
                            if len(train_data) < chunk_size and completed < train_size:
                                train_data.append(it)
                                completed += 1
                                pbar.update(1)
                        
                            if len(train_data) >= chunk_size or completed >= train_size:
                                break
                    if len(train_data) >= chunk_size or completed >= train_size:
                        break
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            if train_data:
                chunk_path = os.path.join(train_dir, f"train_chunk_{chunk_idx:05d}")
                save_chunk_npy(train_data, chunk_path, ctx_len=ctx_len)
                chunk_idx += 1


    merged_train_path = os.path.join(output_dir, "rna_train.npy")
    merge_chunks_npy(train_dir, merged_train_path, ctx_len=ctx_len)
    print(f"[done] merged dataset at: {merged_train_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate RNA dataset using RNAfold (tokenized npy chunks) and merge to a single .npy")
    parser.add_argument("--output_dir", type=str, default="/pvc/dataset/8M0901")
    parser.add_argument("--testraw_path", type=str, default="eterna100_puzzles.tsv",
                        help="TSV with columns 'Secondary Structure V1' and/or 'Secondary Structure V2'")
    parser.add_argument("--train_size", type=int, default=8_000_000)
    parser.add_argument("--chunk_size", type=int, default=10_000)
    parser.add_argument("--min_len", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=400)
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--rnafold_cmd", type=str, default="RNAfold")
    parser.add_argument("--similar_per_seq", type=int, default=9)
    parser.add_argument("--mutate_rate", type=float, default=0.10)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--remove_chunks_after_merge", action="store_true", help="Remove chunk files after successful merge")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.testraw_path, sep="\t")
    v2_structures = set(df["Secondary Structure V2"].dropna().tolist()) if "Secondary Structure V2" in df.columns else set()
    v1_structures = set(df["Secondary Structure V1"].dropna().tolist()) if "Secondary Structure V1" in df.columns else set()
    union_structures = {s for s in (v1_structures | v2_structures) if len(s) <= args.max_len}

    with open(os.path.join(args.output_dir, "rna_test_v2.json"), "wb") as f:
        f.write(orjson.dumps([{"structure": s} for s in v2_structures]))
    with open(os.path.join(args.output_dir, "rna_test_v1.json"), "wb") as f:
        f.write(orjson.dumps([{"structure": s} for s in v1_structures]))

    print(f"[info] Union test structures: {len(union_structures)} (len <= {args.max_len})")

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
        ctx_len=args.ctx_len
    )


    if args.remove_chunks_after_merge:
        chunks_dir = os.path.join(args.output_dir, "train_chunks")
        print(f"[cleanup] removing chunks under {chunks_dir} ...")
        for f in os.listdir(chunks_dir):
            if f.endswith(".npy"):
                try:
                    os.remove(os.path.join(chunks_dir, f))
                except Exception as e:
                    print(f"[warn] failed to remove {f}: {e}")

if __name__ == "__main__":
    main()
