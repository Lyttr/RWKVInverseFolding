import subprocess
import random
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

# ---------- 参数配置 ----------
MIN_LEN = 80
MAX_LEN = 125
TRAIN_SIZE = 2000000
CHUNK_SIZE = 10000
NUM_THREADS = 8
RNAFOLD_CMD = "RNAfold"
OUTPUT_DIR = "/home/gaji/rna_dataset"
TEST_PATH = os.path.join(OUTPUT_DIR, "rna_test.json")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_chunks")
MERGED_TRAIN_PATH = os.path.join(OUTPUT_DIR, "rna_train.json")

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
    return bool(dot and set(dot) != {'.'})

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

# ---------- 数据生成 ----------
def generate_and_validate(test_structures):
    seq = generate_random_rna_sequence(random.randint(MIN_LEN, MAX_LEN))
    dot = run_rnafold(seq)
    if is_valid_structure(dot) and dot not in test_structures:
        return {"sequence": seq, "structure": dot}
    return None

def generate_train_set(test_structures):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    existing_chunks = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith(".json")])
    completed = len(existing_chunks) * CHUNK_SIZE
    print(f"📦 已存在训练集样本数: {completed}，目标总数: {TRAIN_SIZE}")

    chunk_idx = len(existing_chunks)

    with tqdm(total=TRAIN_SIZE, desc="生成训练集", initial=completed) as pbar:
        while completed < TRAIN_SIZE:
            train_data = []
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = [executor.submit(generate_and_validate, test_structures) for _ in range(CHUNK_SIZE * 2)]
                for future in as_completed(futures):
                    item = future.result()
                    if item:
                        train_data.append(item)
                        completed += 1
                        pbar.update(1)
                        if len(train_data) >= CHUNK_SIZE:
                            break

            if train_data:
                path = os.path.join(TRAIN_DIR, f"train_chunk_{chunk_idx:05d}.json")
                save_json(train_data, path)
                print(f"save to train_chunk_{chunk_idx:05d}.json")
                chunk_idx += 1

    print(f"✅ 完成训练集生成，共 {TRAIN_SIZE} 条样本，存储在 {TRAIN_DIR}/")

    # 合并所有 chunk
    merged = []
    print("🔄 正在合并所有训练集分块...")
    for fname in sorted(os.listdir(TRAIN_DIR)):
        if fname.endswith(".json"):
            data = load_json(os.path.join(TRAIN_DIR, fname))
            merged.extend(data)
    save_json(merged, MERGED_TRAIN_PATH)
    print(f"✅ 已合并为 {MERGED_TRAIN_PATH}，共 {len(merged)} 条样本")

# ---------- 主函数 ----------
def main():
    # 加载 eternabench-cm 数据集
    dataset = load_dataset("multimolecule/eternabench-cm")

    # 合并 train 和 test 分区中的结构，过滤长度不超过 125，去重
    structures = set()
    structure_to_sequence = {}
    for split in ["train", "test"]:
        for entry in dataset[split]:
            structure = entry['secondary_structure']
            sequence = entry['sequence']
            if len(structure) <= 125 and structure not in structures:
                structures.add(structure)
                structure_to_sequence[structure] = sequence

    # 构造测试集，保证结构唯一
    test_data = [
        {"sequence": sequence, "structure": structure}
        for structure, sequence in structure_to_sequence.items()
    ]

    # 保存测试集
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(test_data, TEST_PATH)
    print(f"✅ 已保存测试集，共 {len(test_data)} 条样本，路径为 {TEST_PATH}")

    # 生成训练集
    generate_train_set(structures)

if __name__ == "__main__":
    main()
