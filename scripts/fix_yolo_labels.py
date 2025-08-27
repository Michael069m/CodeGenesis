#!/usr/bin/env python3
"""
Fix YOLO label files by remapping class IDs to match a single-class dataset (class 0).

Actions:
- For each .txt file under a labels directory, read each line.
- If the class_id (first token) is not 0, set it to 0.
- Validate there are 5 tokens (class cx cy w h); skip malformed lines.
- Clamp bbox values to [0, 1].
- Optionally run in dry-run mode to preview changes.

Usage:
  python scripts/fix_yolo_labels.py --labels-dir data/train/labels --dry-run
  python scripts/fix_yolo_labels.py --labels-dir data/train/labels
  python scripts/fix_yolo_labels.py --labels-dir data/val/labels
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Stats:
    files_seen: int = 0
    files_modified: int = 0
    lines_seen: int = 0
    lines_kept: int = 0
    class_changes: int = 0
    malformed_lines: int = 0


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def process_file(path: Path, dry_run: bool, stats: Stats) -> None:
    stats.files_seen += 1
    text = path.read_text().strip()
    if not text:
        return

    lines = text.splitlines()
    out_lines: List[str] = []
    file_changed = False

    for line in lines:
        stats.lines_seen += 1
        parts = line.strip().split()
        if len(parts) != 5:
            stats.malformed_lines += 1
            # skip malformed line
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception:
            stats.malformed_lines += 1
            continue

        # Remap classes >=1 to 0 for single-class datasets
        if cls != 0:
            cls = 0
            stats.class_changes += 1
            file_changed = True

        # Clamp coordinates to [0,1]
        x = clamp01(x)
        y = clamp01(y)
        w = clamp01(w)
        h = clamp01(h)

        out_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        stats.lines_kept += 1

    # Write back if changed
    if file_changed and not dry_run:
        if out_lines:
            path.write_text("\n".join(out_lines) + "\n")
        else:
            # If nothing valid remains, write empty file (background image)
            path.write_text("")
        stats.files_modified += 1


def fix_labels(labels_dir: Path, dry_run: bool) -> Stats:
    stats = Stats()
    for path in sorted(labels_dir.rglob("*.txt")):
        process_file(path, dry_run, stats)
    return stats


def main():
    p = argparse.ArgumentParser(description="Fix YOLO labels to single-class (class=0)")
    p.add_argument("--labels-dir", type=Path, required=True, help="Path to labels directory")
    p.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = p.parse_args()

    labels_dir: Path = args.labels_dir
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")

    stats = fix_labels(labels_dir, args.dry_run)
    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(
        f"[{mode}] files_seen={stats.files_seen} files_modified={stats.files_modified} "
        f"lines_seen={stats.lines_seen} lines_kept={stats.lines_kept} "
        f"class_changes={stats.class_changes} malformed_lines={stats.malformed_lines}"
    )


if __name__ == "__main__":
    main()
