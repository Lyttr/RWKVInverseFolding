import matplotlib.pyplot as plt
from collections import Counter
import json
import numpy as np

# Load the JSON data
with open("/home/gaji/rna_dataset/rna_train.json", "r") as f:
    data = json.load(f)

# Extract structures and count occurrences
structures = [entry["structure"] for entry in data]
structure_counts = Counter(structures)
unique_structures = structure_counts.keys()

# Complexity metrics functions
def structure_depth(structure):
    max_depth = depth = 0
    for ch in structure:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    return max_depth

def pairing_span(structure):
    stack = []
    spans = []
    for i, ch in enumerate(structure):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                spans.append(i - j)
    return max(spans) if spans else 0

# Detect if structure contains two or more non-nested parallel base-pair blocks
def has_parallel_blocks(structure):
    stack = []
    pairs = []
    for i, ch in enumerate(structure):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))

    # Sort by opening index
    pairs.sort()

    # Look for any two non-overlapping, non-nested blocks: (a1,b1), (a2,b2) with b1 < a2
    for i in range(len(pairs)):
        a1, b1 = pairs[i]
        for j in range(i + 1, len(pairs)):
            a2, b2 = pairs[j]
            if b1 < a2:
                return True
    return False

# Compute complexity metrics
base_pair_counts = []
max_depths = []
max_spans = []
lengths = []
parallel_block_flags = []

for s in structures:
    base_pair_counts.append(min(s.count('('), s.count(')')))
    max_depths.append(structure_depth(s))
    max_spans.append(pairing_span(s))
    lengths.append(len(s))
    parallel_block_flags.append(has_parallel_blocks(s))

# Output statistics
parallel_count = sum(parallel_block_flags)
print(f"Structures with parallel (non-nested) blocks: {parallel_count} / {len(structures)}")
print(f"Average base pairs: {sum(base_pair_counts) / len(base_pair_counts):.2f}")
print(f"Average max depth: {sum(max_depths) / len(max_depths):.2f}")
print(f"Average max span: {sum(max_spans) / len(max_spans):.2f}")
print(f"Average structure length: {sum(lengths) / len(lengths):.2f}")
print(f"Total structures: {len(structures)}")
print(f"Unique structures: {len(unique_structures)}")

# Plot the cumulative distribution function (CDF) — no log scale
sorted_counts = sorted(structure_counts.values(), reverse=True)
counts_array = np.array(sorted_counts)
cdf = np.cumsum(counts_array) / sum(counts_array)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cdf) + 1), cdf)
# Removed: plt.xscale('log')
plt.xlabel("Top-N Most Frequent Structures")
plt.ylabel("Cumulative Frequency")
plt.title("Cumulative Distribution of RNA Structure Frequencies")
plt.grid(True)
plt.tight_layout()
plt.savefig("structure_cdf.png")
print("CDF plot saved to structure_cdf.png")
plt.show()