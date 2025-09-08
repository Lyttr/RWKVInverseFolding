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
from threading import Lock

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

def mutate_sequence(seq, rate=0.1):
    n = len(seq)
    if n == 0:
        return seq
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
    return True
def load_and_filter_seeds(seed_json_path: str, min_len: int, max_len: int):
    raw = orjson.loads(open(seed_json_path, "rb").read())
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "sequence" in raw[0]:
        seqs_all = [d["sequence"] for d in raw if isinstance(d, dict) and "sequence" in d]
    else:
        seqs_all = [s for s in raw if isinstance(s, str)]

    total = len(seqs_all)
    # 去重再过滤
    seqs_all = list(dict.fromkeys(seqs_all))
    seeds = [s for s in seqs_all if min_len <= len(s) <= max_len]
    print(f"[seed] loaded={total}, dedup={len(seqs_all)}, length-filtered={len(seeds)}, "
          f"range=[{min_len}, {max_len}]")
    if not seeds:
        raise ValueError("No seed sequences remain after min_len/max_len filtering.")
    return seeds
def mutate_one_valid_from_seed(next_seed_fn, rnafold_cmd, test_structures, mutate_rate):
    try:
        seed_seq = next_seed_fn()
        cand = mutate_sequence(seed_seq, rate=mutate_rate)
        dot = run_rnafold(cand, rnafold_cmd)
        if is_valid_structure(dot) and dot not in test_structures:
            return {"sequence": cand, "structure": dot}
    except Exception:
        pass
    return None

def generate_train_set_from_seeds(
    seed_sequences,
    test_structures,
    output_dir,
    train_size,
    chunk_size,
    num_threads,
    rnafold_cmd,
    mutate_rate,
    ctx_len
):
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

    seed_pool = list(seed_sequences)
    random.shuffle(seed_pool)
    seed_lock = Lock()
    seed_idx = 0

    def next_seed():
        nonlocal seed_idx, seed_pool
        with seed_lock:
            if seed_idx >= len(seed_pool):
                random.shuffle(seed_pool)
                seed_idx = 0
            s = seed_pool[seed_idx]
            seed_idx += 1
            return s

    with tqdm(total=train_size, desc="Generating training set", initial=completed) as pbar:
        while completed < train_size:
            remaining_global = train_size - completed
            target_this_chunk = min(chunk_size, remaining_global)

            train_data = []
            while len(train_data) < target_this_chunk and completed < train_size:
                need = target_this_chunk - len(train_data)
                batch_submit = max(int(need * 1.1), num_threads)

                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [
                        executor.submit(
                            mutate_one_valid_from_seed,
                            next_seed,          
                            rnafold_cmd,
                            test_structures,
                            mutate_rate
                        )
                        for _ in range(batch_submit)
                    ]

                    for fut in as_completed(futures):
                        item = fut.result()
                        if item is not None:
                            train_data.append(item)
                            completed += 1
                            pbar.update(1)
                            if len(train_data) >= target_this_chunk or completed >= train_size:
                            
                                for rest in futures:
                                    if not rest.done():
                                        rest.cancel()
                                break

            if train_data:
                chunk_path = os.path.join(train_dir, f"train_chunk_{chunk_idx:05d}")
                save_chunk_npy(train_data, chunk_path, ctx_len=ctx_len)
                chunk_idx += 1

    merged_train_path = os.path.join(output_dir, "rna_train.npy")
    merge_chunks_npy(train_dir, merged_train_path, ctx_len=ctx_len)
    print(f"[done] merged dataset at: {merged_train_path}")

# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate RNA dataset from seed JSON using RNAfold (tokenized npy chunks) and merge to a single .npy"
    )
    parser.add_argument("--output_dir", type=str, default="/pvc/datasets/mutate16M0907")
    parser.add_argument("--testraw_path", type=str, default="eterna100_puzzles.tsv")
    parser.add_argument("--seed_json", type=str, default="/pvc/datasets/seed.json")
    parser.add_argument("--train_size", type=int, default=16000000)
    parser.add_argument("--chunk_size", type=int, default=10_000)
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--rnafold_cmd", type=str, default="RNAfold")
    parser.add_argument("--mutate_rate", type=float, default=0.10)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--min_len", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=400)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.testraw_path, sep="\t")
    v2_structures = set(df["Secondary Structure V2"].dropna().tolist()) if "Secondary Structure V2" in df.columns else set()
    v1_structures = set(df["Secondary Structure V1"].dropna().tolist()) if "Secondary Structure V1" in df.columns else set()
    union_structures = {s for s in (v1_structures | v2_structures) }

    with open(os.path.join(args.output_dir, "rna_test_v2.json"), "wb") as f:
        f.write(orjson.dumps([{"structure": s} for s in v2_structures]))
    with open(os.path.join(args.output_dir, "rna_test_v1.json"), "wb") as f:
        f.write(orjson.dumps([{"structure": s} for s in v1_structures]))

    print(f"[info] Union test structures: {len(union_structures)} ")
    seed_sequences = load_and_filter_seeds(args.seed_json, args.min_len, args.max_len)

    generate_train_set_from_seeds(
        seed_sequences=seed_sequences,
        test_structures=union_structures,
        output_dir=args.output_dir,
        train_size=args.train_size,
        chunk_size=args.chunk_size,
        num_threads=args.num_threads,
        rnafold_cmd=args.rnafold_cmd,
        mutate_rate=args.mutate_rate,
        ctx_len=args.ctx_len
    )

    if True:
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