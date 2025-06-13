import json
import numpy as np
import argparse
import os


def json_to_token_npy(input_json, output_npy, ctx_len=256):
    VOCAB = {
        ".": 0,
        "(": 1,
        ")": 2,
        "A": 3,
        "C": 4,
        "G": 5,
        "U": 6,
        "\n": 7,
        "PAD": 8
    }

    with open(input_json, "r") as f:
        data = json.load(f)

    token_lines = []
    for item in data:
        text = item["structure"] + "\n" + item["sequence"]
        token_ids = [VOCAB[c] for c in text]

        if len(token_ids) < ctx_len:
            token_ids += [pad_id] * (ctx_len - len(token_ids))
        else:
            token_ids = token_ids[:ctx_len]

        token_lines.append(token_ids)

    tokens_np = np.array(token_lines, dtype=np.uint16)  # shape [B, T]
    np.save(output_npy, tokens_np)

    print(f"Total samples: {len(token_lines)}")
    print(f"Each line shape: ({ctx_len},)")
    print(f"Final array shape: {tokens_np.shape} (B, T)")
    print(f"Output saved to: {output_npy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RNA JSON to tokenized NPY array")
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--output_npy", required=True, help="Path to output .npy file")
    parser.add_argument("--ctx_len", type=int, default=256, help="Context length (default: 256)")
  
    args = parser.parse_args()

    json_to_token_npy(
        input_json=args.input_json,
        output_npy=args.output_npy,
        ctx_len=args.ctx_len,
      
    )