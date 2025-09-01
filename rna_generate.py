import subprocess
import random
import json
import os
from tqdm import tqdm
import argparse
import pandas as pd
import orjson
import shutil
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

def save_json(data, path):
    with open(path, "wb") as f:
        f.write(orjson.dumps(data))

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _extract_dot(token: str):
    return token if set(token) <= set(".()") else None

def fold_batch_rnafold(seqs, rnafold_cmd="RNAfold", jobs=0, timeout=None):
    if not seqs:
        return []
    inp = "\n".join(seqs) + "\n"

    def _run(cmd):
        try:
            return subprocess.run(
                cmd, input=inp, text=True, capture_output=True,
                timeout=timeout, check=False
            )
        except Exception:
            return None

    cmd = [rnafold_cmd, "--noPS"]
    if jobs is not None:
        cmd += ["-j", str(jobs)]  
    res = _run(cmd)
    if res is None or res.returncode != 0:
        res = _run([rnafold_cmd, "--noPS"])
        if res is None or res.returncode != 0:
            return [None] * len(seqs)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    dots = []
    i = 0
    while i < len(lines):
       
        if i + 1 < len(lines):
            tok = (lines[i + 1].split() or [""])[0]
            dot = _extract_dot(tok)
            dots.append(dot)
            i += 2
        else:
            dots.append(None)
            i += 1
    if len(dots) < len(seqs):
        dots += [None] * (len(seqs) - len(dots))
    elif len(dots) > len(seqs):
        dots = dots[:len(seqs)]
    return dots

def fold_batch(seqs, rnafold_cmd="RNAfold", jobs=0, timeout=None):

    if shutil.which(rnafold_cmd) is None:
        return [None] * len(seqs)
    return fold_batch_rnafold(seqs, rnafold_cmd, jobs=jobs, timeout=timeout)
def generate_train_set(
    test_structures,        
    output_dir,
    train_size,
    chunk_size,
    num_threads,           
    min_len,
    max_len,
    rnafold_cmd,
    similar_per_seq=99,
    mutate_rate=0.10,
    batch_bases=256,         
    fold_jobs=0,             
    fold_timeout=None        
):
    train_dir = os.path.join(output_dir, "train_chunks")
    os.makedirs(train_dir, exist_ok=True)
    existing_chunks = sorted([f for f in os.listdir(train_dir) if f.endswith(".json")])
    completed = len(existing_chunks) * chunk_size
    print(f"Completed: {completed}, Target: {train_size}")

    chunk_idx = len(existing_chunks)

    def rand_base():
        return generate_random_rna_sequence(random.randint(min_len, max_len))

    with tqdm(total=train_size, desc="Generating training set", initial=completed) as pbar:
        while completed < train_size:
            train_data = []

            while len(train_data) < chunk_size and completed < train_size:
            
                base_seqs = [rand_base() for _ in range(batch_bases)]
                cand_seqs = []
                cand_seqs.extend(base_seqs)
                for s in base_seqs:
                    for _ in range(similar_per_seq):
                        cand_seqs.append(mutate_sequence(s, mutate_rate))

                cand_dots = fold_batch(
                    cand_seqs,
                    rnafold_cmd=rnafold_cmd,
                    jobs=fold_jobs,
                    timeout=fold_timeout
                )

               
                for seq, dot in zip(cand_seqs, cand_dots):
                    if dot is None:
                        continue
                    
                    if dot in test_structures:
                        continue
                    train_data.append({"sequence": seq, "structure": dot})
                    completed += 1
                    pbar.update(1)
                    if len(train_data) >= chunk_size or completed >= train_size:
                        break

          
            if train_data:
                if len(train_data) > chunk_size:
                    train_data = train_data[:chunk_size]
                path = os.path.join(train_dir, f"train_chunk_{chunk_idx:05d}.json")
                save_json(train_data, path)
                print(f"Saved chunk to {path} (size={len(train_data)})")
                chunk_idx += 1

    print(f"Training set generated: {train_size} samples, saved to {train_dir}/")
    merged = []
    for fname in sorted(os.listdir(train_dir)):
        if fname.endswith(".json"):
            data = load_json(os.path.join(train_dir, fname))
            merged.extend(data)
    merged_train_path = os.path.join(output_dir, "rna_train.json")
    save_json(merged, merged_train_path)
    print(f"Chunks merged to {merged_train_path}, total: {len(merged)} samples")

def main():
    parser = argparse.ArgumentParser(description="Generate RNA dataset using RNAfold with similar-sequence augmentation (batched)")
    parser.add_argument("--output_dir", type=str, default="/pvc/dataset/8M0831")
    parser.add_argument("--testraw_dir", type=str, default="eterna100_puzzles.tsv")
    parser.add_argument("--train_size", type=int, default=8000000)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--min_len", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=400)
    parser.add_argument("--num_threads", type=int, default=8)   
    parser.add_argument("--rnafold_cmd", type=str, default="RNAfold")
    parser.add_argument("--similar_per_seq", type=int, default=99)
    parser.add_argument("--mutate_rate", type=float, default=0.10)
    parser.add_argument("--batch_bases", type=int, default=256)
    parser.add_argument("--fold_jobs", type=int, default=0)
    parser.add_argument("--fold_timeout", type=int, default=None)
    args = parser.parse_args()

    test_v2_path_compat = os.path.join(args.output_dir, "rna_test.json")     
    test_v1_path = os.path.join(args.output_dir, "rna_test_v1.json")        

    df = pd.read_csv(args.testraw_dir, sep="\t")
    v2_structures = set()
    if "Secondary Structure V2" in df.columns:
        for structure in df["Secondary Structure V2"].dropna():
            if len(structure) <= args.max_len:
                v2_structures.add(structure)
    v2_test_data = [{"structure": s} for s in v2_structures]
    v1_structures = set()
    if "Secondary Structure V1" in df.columns:
        for structure in df["Secondary Structure V1"].dropna():
            if len(structure) <= args.max_len:
                v1_structures.add(structure)
    v1_test_data = [{"structure": s} for s in v1_structures]
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(v2_test_data, test_v2_path_compat)
    print(f"Test set (V2) saved: {len(v2_test_data)} structures → {test_v2_path_compat}")
    save_json(v1_test_data, test_v1_path)
    print(f"Test set (V1) saved: {len(v1_test_data)} structures → {test_v1_path}")
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
        fold_jobs=args.fold_jobs,
        fold_timeout=args.fold_timeout
    )

if __name__ == "__main__":
    main()
