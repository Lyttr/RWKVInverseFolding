import json
import numpy as np
import os

# ---------- 配置 ----------
INPUT_JSON = "/home/gaji/rna_dataset/rna_train.json"
OUTPUT_NPY = "/home/gaji/rna_dataset/rna_train_token_lines.npy"
CTX_LEN = 256
PAD_ID = 8
VOCAB = {
    ".": 0,
    "(": 1,
    ")": 2,
    "A": 3,
    "C": 4,
    "G": 5,
    "U": 6,
    "\n": 7,
    "PAD": PAD_ID
}

# ---------- 加载数据 ----------
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# ---------- Tokenize per pair ----------
token_lines = []
for item in data:
    text = item["structure"] + "\n" + item["sequence"]
    token_ids = [VOCAB[c] for c in text]

    if len(token_ids) < CTX_LEN:
        token_ids += [PAD_ID] * (CTX_LEN - len(token_ids))
    else:
        token_ids = token_ids[:CTX_LEN]

    token_lines.append(token_ids)

# ---------- 保存 ----------
tokens_np = np.array(token_lines, dtype=np.uint16)  # shape [B, T]
np.save(OUTPUT_NPY, tokens_np)

# ---------- 输出信息 ----------
print(f"✅ Total samples: {len(token_lines)}")
print(f"✅ Each line shape: ({CTX_LEN},)")
print(f"✅ Final array shape: {tokens_np.shape} (B, T)")
print(f"📂 Output saved to: {OUTPUT_NPY}")