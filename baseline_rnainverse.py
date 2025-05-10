import subprocess
import json

def run_rnainverse(structure):
    cmd = ["RNAinverse"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    input_data = f"{structure}\n"
    stdout, _ = p.communicate(input=input_data)
    lines = stdout.strip().split('\n')
    
    return lines[0].split()[0]
    

def run_rnafold(sequence):
    cmd = ["RNAfold", "--noPS"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, _ = p.communicate(input=sequence + "\n")
    lines = stdout.strip().split('\n')
    if len(lines) >= 2:
        structure = lines[1].split()[0]
        return structure
    return None

def compute_metrics(predicted, target):
    correct = sum(1 for a, b in zip(predicted, target) if a == b)
    correct_rate = correct / len(target)
    edit_distance = sum(1 for a, b in zip(predicted, target) if a != b) + abs(len(predicted) - len(target))
    full_match = int(predicted == target)
    return correct_rate, edit_distance, full_match

def process_jsonl(input_path, output_jsonl_path="rna_inverse_results.jsonl", summary_path="summary.json"):
    output = []
    total_correct_rate = 0
    total_edit_distance = 0
    total_full_match = 0
    count = 0

    with open(input_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            target_struct = obj["target_structure"]

            # 1. RNAinverse
            gen_seq = run_rnainverse(target_struct)
            if not gen_seq:
                print("RNAinverse failed for:", target_struct)
                continue

            # 2. RNAfold
            folded_struct = run_rnafold(gen_seq)
            if not folded_struct:
                print("RNAfold failed for:", gen_seq)
                continue

            # 3. Metrics
            correct_rate, edit_dist, full_match = compute_metrics(folded_struct, target_struct)

            result = {
                "predicted_sequence": gen_seq,
                "predicted_structure": folded_struct,
                "target_structure": target_struct,
                "correct_rate": round(correct_rate, 3),
                "edit_distance": edit_dist,
                "full_match": full_match
            }
            output.append(result)

            total_correct_rate += correct_rate
            total_edit_distance += edit_dist
            total_full_match += full_match
            count += 1

    # Save per-sample results to jsonl
    with open(output_jsonl_path, 'w') as out_file:
        for item in output:
            out_file.write(json.dumps(item) + "\n")

    # Save summary statistics
    summary = {}
    if count > 0:
        summary = {
            "average_correct_rate": round(total_correct_rate / count, 3),
            "average_edit_distance": round(total_edit_distance / count, 3),
            "average_full_match": round(total_full_match / count, 3),
            "total_samples": count
        }
    else:
        summary = {
            "message": "No valid samples processed."
        }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("Results saved to:", output_jsonl_path)
    print("Summary saved to:", summary_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python rna_inverse_eval.py input.jsonl")
    else:
        process_jsonl(sys.argv[1])