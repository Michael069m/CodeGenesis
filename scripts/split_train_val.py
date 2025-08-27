#!/usr/bin/env python3
"""Split a portion of train into val, preserving YOLO image/label pairs.

Usage examples:
  python3 scripts/split_train_val.py --data-dir ml-training-project/data --val-ratio 0.1
  python3 scripts/split_train_val.py --data-dir ml-training-project/data --val-ratio 0.2 --copy --dry-run
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    for img_path in images_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in ALLOWED_IMG_EXTS:
            continue
        # 1) Try direct match: <stem>.txt
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
            continue
        # 2) Fallback: if image stem starts with 'image_' but labels use 'label_' prefix
        stem = img_path.stem
        if stem.startswith("image_"):
            suffix = stem.split("_", 1)[1]
            alt_lbl = labels_dir / (f"label_{suffix}.txt")
            if alt_lbl.exists():
                pairs.append((img_path, alt_lbl))
                continue
        # 3) If no match, warn
        print(f"[warn] Missing label for image: {img_path.name}")
    return pairs


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description="Split a portion of train into val, preserving YOLO pairs.")
    ap.add_argument("--data-dir", default="data", type=str, help="Root data dir containing train/ and val/")
    ap.add_argument("--train-subdir", default="train", type=str)
    ap.add_argument("--val-subdir", default="val", type=str)
    ap.add_argument("--images-subdir", default="images", type=str)
    ap.add_argument("--labels-subdir", default="labels", type=str)
    ap.add_argument("--val-ratio", default=0.1, type=float, help="Fraction of pairs to move/copy to val")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--copy", action="store_true", help="Copy instead of move (default: move)")
    ap.add_argument("--dry-run", action="store_true", help="Show planned ops without changing files")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_images = data_dir / args.train_subdir / args.images_subdir
    train_labels = data_dir / args.train_subdir / args.labels_subdir
    val_images = data_dir / args.val_subdir / args.images_subdir
    val_labels = data_dir / args.val_subdir / args.labels_subdir

    ensure_dirs(val_images, val_labels)

    pairs = find_pairs(train_images, train_labels)
    n_total = len(pairs)
    if n_total == 0:
        print("[error] No (image,label) pairs found in train.")
        return

    k = int(n_total * args.val_ratio)
    k = max(k, 1)
    random.seed(args.seed)
    random.shuffle(pairs)
    val_samples = pairs[:k]

    op = "COPY" if args.copy else "MOVE"
    print(f"[info] Found {n_total} pairs in train. {op} {k} pairs to val ({args.val_ratio*100:.1f}%).")
    if args.dry_run:
        for img_path, _ in val_samples:
            print(f"[dry-run] {op} {img_path.name}")
        return

    for img_path, lbl_path in val_samples:
        dst_img = val_images / img_path.name
        dst_lbl = val_labels / lbl_path.name
        move_or_copy(img_path, dst_img, args.copy)
        move_or_copy(lbl_path, dst_lbl, args.copy)

    # Report counts
    new_train_pairs = find_pairs(train_images, train_labels)
    new_val_pairs = find_pairs(val_images, val_labels)
    print(f"[done] Train pairs: {len(new_train_pairs)} | Val pairs: {len(new_val_pairs)}")


if __name__ == "__main__":
    main()