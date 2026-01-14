#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


def parse_video_id(filename: str) -> str:
    """
    Extract the original video ID from a filename like:
    'ab28GAufK8o#000261#000596.mp4' -> 'ab28GAufK8o'
    """
    stem = Path(filename).stem
    parts = stem.split("#")
    return parts[0] if parts else stem


def build_pairs(files, max_pairs=None, seed=42):
    """
    Build (source, driving) pairs such that:
      - source != driving
      - they come from different original videos
    Returns a list of (source_name, driving_name).
    """
    basenames = [f.name for f in files]
    video_ids = [parse_video_id(name) for name in basenames]

    candidate_pairs = []
    n = len(basenames)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if video_ids[i] == video_ids[j]:
                continue  # skip same original video
            candidate_pairs.append((basenames[i], basenames[j]))

    if not candidate_pairs:
        return []

    if max_pairs is not None and max_pairs < len(candidate_pairs):
        random.seed(seed)
        candidate_pairs = random.sample(candidate_pairs, max_pairs)

    return candidate_pairs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a taichi-style pairs CSV from a directory of .mp4 clips.\n"
            "Columns: distance, source, driving, frame.\n"
            "Clips from the same original video (same prefix before '#') "
            "will NOT be paired together."
        )
    )
    parser.add_argument(
        "--clips_dir",
        type=Path,
        required=True,
        help="Directory containing all .mp4 clips.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Path to output CSV (e.g. my_pairs.csv).",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to generate (optional). "
             "If not set, all valid pairs are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling pairs (if --max_pairs is set).",
    )

    args = parser.parse_args()

    if not args.clips_dir.is_dir():
        raise SystemExit(f"clips_dir is not a directory: {args.clips_dir}")

    files = sorted(args.clips_dir.glob("*.mp4"))
    if len(files) < 2:
        raise SystemExit("Need at least 2 .mp4 files to form pairs.")

    pairs = build_pairs(files, max_pairs=args.max_pairs, seed=args.seed)
    if not pairs:
        raise SystemExit(
            "No valid pairs found (all clips may come from the same original video?)."
        )

    # Write CSV: distance, source, driving, frame
    with args.out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "source", "driving", "frame"])
        for source_name, driving_name in pairs:
            distance = 0.0
            frame = 0
            writer.writerow([distance, source_name, driving_name, frame])

    print(f"Wrote {len(pairs)} pairs to {args.out_csv}")


if __name__ == "__main__":
    main()
