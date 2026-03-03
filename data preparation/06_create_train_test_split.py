"""
Step 5: Create train/test split from image_text_mapping.json.

Input:
    - data/image_text_mapping.json — full mapping from step 4

Output:
    - data/train_lines.json — training pairs (~550)
    - data/test_lines.json  — test pairs (~200)

Split strategy:
    - Flatten ALL pairs from ALL sources into one list
    - Random shuffle with seed=42 for reproducibility
    - First 200 pairs → test set
    - Remaining pairs → train set

Note on manual verification:
    After this split, test pairs were manually reviewed.
    Low quality or misaligned crops removed.
    Final verified test set saved as accept_test_lines.json (~195 pairs).

Usage:
    python 05_create_train_test_split.py \
        --mapping data/image_text_mapping.json \
        --output_dir data \
        --test_size 200 \
        --seed 42
"""

import os
import json
import random
import argparse

# --- CONFIG ---
DEFAULT_MAPPING_PATH = "data/image_text_mapping.json"
DEFAULT_OUTPUT_DIR   = "data"
DEFAULT_TEST_SIZE    = 200
DEFAULT_SEED         = 42


def flatten_pairs(mapping):
    """
    Flatten all pairs from all source/page keys into one list.

    Args:
        mapping: Dict loaded from image_text_mapping.json

    Returns:
        Flat list of all pair dicts
    """
    all_pairs = []
    for page_key, page_data in mapping.items():
        for pair in page_data['pairs']:
            all_pairs.append(pair)
    return all_pairs


def split_train_test(all_pairs, test_size=200, seed=42):
    """
    Randomly shuffle and split into train/test.

    Strategy:
        - All pairs from all sources pooled together
        - Random shuffle with fixed seed for reproducibility
        - First test_size pairs → test
        - Remaining → train

    Args:
        all_pairs: Flat list of all pairs
        test_size: Number of pairs for test set
        seed: Random seed

    Returns:
        train_pairs, test_pairs
    """
    random.seed(seed)
    random.shuffle(all_pairs)

    test_pairs  = all_pairs[:test_size]
    train_pairs = all_pairs[test_size:]

    return train_pairs, test_pairs


def save_json(data, path):
    """Save list as formatted JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path} ({len(data)} pairs)")


def print_source_breakdown(train_pairs, test_pairs):
    """Print per-source pair counts for verification."""
    source_counts = {}

    for pairs, split in [(train_pairs, 'train'), (test_pairs, 'test')]:
        for p in pairs:
            src = p['source']
            if src not in source_counts:
                source_counts[src] = {'train': 0, 'test': 0}
            source_counts[src][split] += 1

    print(f"\nPer-source breakdown:")
    print(f"{'Source':<45} {'Train':>6} {'Test':>6} {'Total':>7}")
    print("-" * 65)
    for src, counts in sorted(source_counts.items()):
        total = counts['train'] + counts['test']
        print(
            f"{src:<45} "
            f"{counts['train']:>6} "
            f"{counts['test']:>6} "
            f"{total:>7}"
        )


def run(mapping_path, output_dir, test_size, seed):
    """
    Run train/test split and save.

    Args:
        mapping_path: Path to image_text_mapping.json
        output_dir: Where to save train_lines.json and test_lines.json
        test_size: Number of pairs for test set
        seed: Random seed for reproducibility
    """
    print(f"Loading mapping from: {mapping_path}")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # Step 1 — Flatten
    all_pairs = flatten_pairs(mapping)
    print(f"Total pairs: {len(all_pairs)}")

    # Step 2 — Split
    train_pairs, test_pairs = split_train_test(
        all_pairs,
        test_size=test_size,
        seed=seed
    )

    # Step 3 — Save
    train_path = os.path.join(output_dir, "train_lines.json")
    test_path  = os.path.join(output_dir, "test_lines.json")

    save_json(train_pairs, train_path)
    save_json(test_pairs,  test_path)

    # Summary
    print(f"\n{'='*50}")
    print(f"Total pairs:  {len(all_pairs)}")
    print(f"Train pairs:  {len(train_pairs)}")
    print(f"Test pairs:   {len(test_pairs)}")
    print(f"Seed:         {seed}")
    print(f"{'='*50}")

    print_source_breakdown(train_pairs, test_pairs)

    print(f"\nNote: test_lines.json was manually verified after this split.")
    print(f"Final verified test set saved as accept_test_lines.json (~195 pairs).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/test split from image_text_mapping.json"
    )
    parser.add_argument(
        "--mapping",
        default=DEFAULT_MAPPING_PATH,
        help="Path to image_text_mapping.json"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for train/test JSON files"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help="Number of pairs for test set (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    run(
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed
    )