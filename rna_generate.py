import subprocess
import random
import json
import os
from tqdm import tqdm
from collections import defaultdict

# ---------- 参数配置 ----------
MIN_LEN = 80
MAX_LEN = 120
TOTAL_STRUCTURES = 2000000
TRAIN_RATIO = 0.9
RNAFOLD_CMD = "RNAfold"
CHUNK_SIZE = 10000
OUTPUT_DIR = "rna_dataset/rna_generate"

# ---------- 工具函数 ----------
def generate_random_rna_sequence(n):
    return ''.join(random.choices('AUGC', k=n))

def run_rnafold(seq):
    try:
        result = subprocess.run(
            [RNAFOLD_CMD, "--noPS"],
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
    if not dot:
        return False
    if set(dot) == {'.'}:
        return False
    return True

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------- 主生成函数 ----------
def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = []
    count = 0
    tries = 0

    with tqdm(total=TOTAL_STRUCTURES, desc="Generating", miniters=1000) as pbar:
        while count < TOTAL_STRUCTURES and tries < TOTAL_STRUCTURES * 100:
            tries += 1
            length = random.randint(MIN_LEN, MAX_LEN)
            seq = generate_random_rna_sequence(length)
            dot = run_rnafold(seq)

            if dot and is_valid_structure(dot):
                dataset.append({"structure": dot, "sequence": seq})
                count += 1
                pbar.update(1)

    print(f"✅ Finished generation: {count} samples after {tries} tries")
    return dataset

# ---------- 数据划分函数（结构级不重复） ----------
def split_and_save(dataset):
    struct_to_items = defaultdict(list)
    for item in dataset:
        struct_to_items[item["structure"]].append(item)

    all_structs = list(struct_to_items.keys())
    random.shuffle(all_structs)

    cutoff = int(len(all_structs) * TRAIN_RATIO)
    train_structs = set(all_structs[:cutoff])
    test_structs = set(all_structs[cutoff:])

    train = []
    test = []

    for s in train_structs:
        train.extend(struct_to_items[s])
    for s in test_structs:
        test.extend(struct_to_items[s])

    train_path = os.path.join(OUTPUT_DIR, "rna_train.json")
    test_path = os.path.join(OUTPUT_DIR, "rna_test.json")
    save_json(train, train_path)
    save_json(test, test_path)

    print(f"✅ Train samples: {len(train)} | Test samples: {len(test)}")
    print(f"✅ Unique structures: Train: {len(train_structs)} | Test: {len(test_structs)}")
    print(f"📂 Saved to: {train_path}, {test_path}")

# ---------- 主函数 ----------
def main():
    print(f"Generating {TOTAL_STRUCTURES} structure-sequence pairs into: {OUTPUT_DIR}")
    dataset = generate_dataset()
    split_and_save(dataset)

if __name__ == "__main__":
    main()
