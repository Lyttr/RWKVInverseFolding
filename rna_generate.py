#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
import shutil
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import orjson

VOCAB = {
    ".": 0, "(": 1, ")": 2,
    "A": 3, "C": 4, "G": 5, "U": 6,
    "\n": 7, "PAD": 8
}

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
    print(f"Saved chunk (npy) to {path}.npy, shape={arr.shape}")


def merge_chunks_npy(train_dir, output_path, ctx_len=256):
    files = sorted([f for f in os.listdir(train_dir) if f.endswith(".npy")])
    arrays = []
    total = 0
    for f in files:
        arr = np.load(os.path.join(train_dir, f))
        arrays.append(arr)
        total += arr.shape[0]
    merged = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(total, ctx_len))
    offset = 0
    for arr in arrays:
        merged[offset:offset+arr.shape[0]] = arr
        offset += arr.shape[0]
    merged.flush()
    print(f"Merged {len(files)} chunks → {output_path}, shape={merged.shape}")
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
    except:
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
    completed = len(existing_chunks) * chunk_size
    print(f"Completed: {completed}, Target: {train_size}")
    chunk_idx = len(existing_chunks)

    with tqdm(total=train_size, desc="Generating training set", initial=completed) as pbar:
        while completed < train_size:
            train_data = []
            executor = ThreadPoolExecutor(max_workers=num_threads)
            futures = [
                executor.submit(
                    generate_and_validate,
                    test_structures, min_len, max_len,
                    rnafold_cmd, similar_per_seq, mutate_rate
                )
                for _ in range((int)(chunk_size * 1.1)) 
            ]

            try:
                for future in as_completed(futures):
                    items = future.result()
                    if items:
                        train_data.extend(items)
                        for _ in range(len(items)):
                            completed += 1
                            pbar.update(1)
                            if len(train_data) >= chunk_size or completed >= train_size:
                                break
                    if len(train_data) >= chunk_size or completed >= train_size:
                        break
            finally:
               
                executor.shutdown(wait=False, cancel_futures=True)

            if train_data:
                if len(train_data) > chunk_size:
                    train_data = train_data[:chunk_size]
                chunk_path = os.path.join(train_dir, f"train_chunk_{chunk_idx:05d}")
                save_chunk_npy(train_data, chunk_path, ctx_len=ctx_len)
                chunk_idx += 1

    merged_train_path = os.path.join(output_dir, "rna_train.npy")
    merge_chunks_npy(train_dir, merged_train_path, ctx_len=ctx_len)
def main():
    parser = argparse.ArgumentParser(description="Generate RNA dataset using RNAfold (tokenized npy chunks)")
    parser.add_argument("--output_dir", type=str, default="/pvc/dataset/8M0901")
    parser.add_argument("--testraw_dir", type=str, default="eterna100_puzzles.tsv")
    parser.add_argument("--train_size", type=int, default=8000000)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--min_len", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=400)
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--rnafold_cmd", type=str, default="RNAfold")
    parser.add_argument("--similar_per_seq", type=int, default=9)
    parser.add_argument("--mutate_rate", type=float, default=0.10)
    parser.add_argument("--ctx_len", type=int, default=1024)
    args = parser.parse_args()
    df = pd.read_csv(args.testraw_dir, sep="\t")
    v2_structures = set(df["Secondary Structure V2"].dropna().tolist()) if "Secondary Structure V2" in df.columns else set()
    v1_structures = set(df["Secondary Structure V1"].dropna().tolist()) if "Secondary Structure V1" in df.columns else set()
    union_structures = {s for s in (v1_structures | v2_structures) if len(s) <= args.max_len}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "rna_test.json"), "wb") as f:
        f.write(orjson.dumps([{"structure": s} for s in v2_structures]))
    with open(os.path.join(args.output_dir, "rna_test_v1.json"), "wb") as f:
        f.write(orjson.dumps([{"structure": s} for s in v1_structures]))
    print(f"Union test structures: {len(union_structures)}")

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

if __name__ == "__main__":
    main()
