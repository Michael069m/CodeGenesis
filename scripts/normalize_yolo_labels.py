#!/usr/bin/env python3
"""
Normalize YOLO label filenames so each image stem has a matching <stem>.txt label.
Handles cases where labels are named like 'label_<suffix>.txt' while images are 'image_<suffix>.<ext>'.

Usage:
  python3 scripts/normalize_yolo_labels.py --images-dir data/train/images --labels-dir data/train/labels
  python3 scripts/normalize_yolo_labels.py --images-dir data/val/images --labels-dir data/val/labels
"""
from pathlib import Path
import argparse
import shutil

ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def normalize(images_dir: Path, labels_dir: Path, dry_run: bool = False) -> int:
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing images or labels dir: {images_dir} | {labels_dir}")
    changes = 0
    for img in images_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in ALLOWED_IMG_EXTS:
            continue
        target_lbl = labels_dir / f"{img.stem}.txt"
        if target_lbl.exists():
            continue
        # Try to locate alternative label with 'label_' prefix
        suffix = img.stem.split("_", 1)[1] if "_" in img.stem else img.stem
        alt_lbl = labels_dir / f"label_{suffix}.txt"
        if alt_lbl.exists():
            print(f"[fix] {alt_lbl.name} -> {target_lbl.name}")
            if not dry_run:
                shutil.move(str(alt_lbl), str(target_lbl))
            changes += 1
        else:
            print(f"[warn] No label found for image: {img.name}")
    return changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    imgs = Path(args.images_dir)
    lbls = Path(args.labels_dir)
    changed = normalize(imgs, lbls, args.dry_run)
    print(f"[done] Renamed {changed} label files to match image stems.")

if __name__ == "__main__":
    main()
