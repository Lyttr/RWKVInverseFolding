import json
import numpy as np
import argparse
import os
import tempfile


def json_to_token_npy(input_json, output_npy, ctx_len=256):
    VOCAB = {
        ".": 0, "(": 1, ")": 2,
        "A": 3, "C": 4, "G": 5, "U": 6,
        "\n": 7, "PAD": 8
    }

    with open(input_json, "r") as f:
        data = json.load(f)

    num_samples = len(data)
    print(f"Total samples: {num_samples}")

    # 用临时文件名作为 .dat 写入位置
    temp_dat_path = output_npy + ".dat"

    # 创建 memmap：写入时不会占用太多内存
    mmap_array = np.memmap(temp_dat_path, dtype=np.uint16, mode='w+', shape=(num_samples, ctx_len))

    for i, item in enumerate(data):
        text = item["structure"] + "\n" + item["sequence"]
        token_ids = [VOCAB[c] for c in text]

        if len(token_ids) < ctx_len:
            token_ids += [VOCAB["PAD"]] * (ctx_len - len(token_ids))
        else:
            token_ids = token_ids[:ctx_len]

        mmap_array[i] = token_ids

        if i % 10000 == 0:
            print(f"Processed {i}/{num_samples}")

    mmap_array.flush()

    # === 将内存映射内容复制为标准 .npy 文件 ===
    np.save(output_npy, np.array(mmap_array))  # 触发写入标准格式
    print(f"Saved to: {output_npy}.npy")

    # 可选：删除临时 dat 文件
    os.remove(temp_dat_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RNA JSON to tokenized NPY array")
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--output_npy", required=True, help="Output .npy file (without .npy extension)")
    parser.add_argument("--ctx_len", type=int, default=256, help="Max token length per sample")

    args = parser.parse_args()

    json_to_token_npy(
        input_json=args.input_json,
        output_npy=args.output_npy,
        ctx_len=args.ctx_len,
    )