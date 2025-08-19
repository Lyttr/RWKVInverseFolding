import subprocess
import random
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
import argparse
import pandas as pd

def generate_random_rna_sequence(n):
    return ''.join(random.choices('AUGC', k=n))


def run_rnafold(seq, rnafold_cmd):
    try:
        result = subprocess.run(
            [rnafold_cmd, "--noPS"],
            input=seq + "\n",
            capture_output=True,
            text=True,
            timeout=3
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

def is_valid_structure(dot):
    return bool(dot and set(dot) != {'.'})
import orjson
def save_json(data, path):
    with open(path, "wb") as f:
        f.write(orjson.dumps(data))
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def generate_and_validate(test_structures, min_len, max_len, rnafold_cmd):
    seq = generate_random_rna_sequence(random.randint(min_len, max_len))
    dot = run_rnafold(seq, rnafold_cmd)
    if is_valid_structure(dot) and dot not in test_structures:
        return {"sequence": seq, "structure": dot}
    return None


def generate_train_set(test_structures, output_dir, train_size, chunk_size, num_threads, min_len, max_len, rnafold_cmd):
    train_dir = os.path.join(output_dir, "train_chunks")
    os.makedirs(train_dir, exist_ok=True)
    existing_chunks = sorted([f for f in os.listdir(train_dir) if f.endswith(".json")])
    completed = len(existing_chunks) * chunk_size
    print(f"Completed: {completed}, Target: {train_size}")

    chunk_idx = len(existing_chunks)

    with tqdm(total=train_size, desc="Generating training set", initial=completed) as pbar:
        while completed < train_size:
            train_data = []
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(generate_and_validate, test_structures, min_len, max_len, rnafold_cmd)
                    for _ in range(chunk_size * 2)
                ]
                for future in as_completed(futures):
                    item = future.result()
                    if item:
                        train_data.append(item)
                        completed += 1
                        pbar.update(1)
                        if len(train_data) >= chunk_size:
                            break

            if train_data:
                path = os.path.join(train_dir, f"train_chunk_{chunk_idx:05d}.json")
                save_json(train_data, path)
                print(f"Saved chunk to {path}")
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
    parser = argparse.ArgumentParser(description="Generate RNA dataset using RNAfold")
    parser.add_argument("--output_dir", type=str, default="/pvc/dataset/8M")
    parser.add_argument("--testraw_dir", type=str, default="eterna100_puzzles.tsv")
    parser.add_argument("--train_size", type=int, default=8000000)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--min_len", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=400)
    parser.add_argument("--num_threads", type=int, default=24)
    parser.add_argument("--rnafold_cmd", type=str, default="RNAfold")
    args = parser.parse_args()

    test_path = os.path.join(args.output_dir, "rna_test.json")

  
    tsv_path = args.testraw_dir
  
 
   


    df = pd.read_csv(tsv_path, sep="\t")

    structures = set()
    for structure in df["Secondary Structure V2"].dropna():
        if len(structure) <=  args.max_len:
            structures.add(structure)
   
  
    test_data = [{"structure": structure} for structure in structures]

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(test_data, test_path)
    print(f"Test set saved: {len(test_data)} structures → {test_path}")

    generate_train_set(
        test_structures=structures,
        output_dir=args.output_dir,
        train_size=args.train_size,
        chunk_size=args.chunk_size,
        num_threads=args.num_threads,
        min_len=args.min_len,
        max_len=args.max_len,
        rnafold_cmd=args.rnafold_cmd
    )


if __name__ == "__main__":
    main()
