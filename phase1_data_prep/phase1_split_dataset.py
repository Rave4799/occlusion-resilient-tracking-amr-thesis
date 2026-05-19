#!/usr/bin/env python3
"""Split labeled images into train/val/test sets with 70/15/15 ratio.

Only images that have a matching .txt label in the labels folder are included.
Copies images and labels into target directories and prints counts.
"""

import argparse
from pathlib import Path
import random
import shutil
from typing import List, Tuple

# Base directory for workspace
BASE_DIR = Path(__file__).resolve().parent.parent

# default directories used by the thesis workspace
DEFAULT_IMAGES_DIR = BASE_DIR / 'dataset' / 'train' / 'images_selected'
DEFAULT_LABELS_DIR = BASE_DIR / 'dataset' / 'train' / 'labels'

# output locations for splits
OUT_TRAIN_IMAGES = BASE_DIR / 'dataset' / 'train' / 'images'
OUT_VAL_IMAGES = BASE_DIR / 'dataset' / 'val' / 'images'
OUT_TEST_IMAGES = BASE_DIR / 'dataset' / 'test' / 'images'

OUT_TRAIN_LABELS = BASE_DIR / 'dataset' / 'train' / 'labels_split'
OUT_VAL_LABELS = BASE_DIR / 'dataset' / 'val' / 'labels'
OUT_TEST_LABELS = BASE_DIR / 'dataset' / 'test' / 'labels'

RATIOS = (0.70, 0.15, 0.15)  # train/val/test


def list_images_with_labels(images_dir: Path, labels_dir: Path) -> List[str]:
    """Return a list of image filenames (basename) that have matching label files.

    Images considered: .jpg, .jpeg, .png (case-insensitive). Matching label must be same basename + '.txt'.
    """
    imgs = [p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    matched = []  # will hold basenames that have labels
    for im in imgs:
        base = Path(im).stem  # remove extension
        label_path = labels_dir / (base + '.txt')  # expected label path
        if label_path.is_file():  # include only if label exists
            matched.append(im)
    return sorted(matched)


def make_dirs(paths: List[Path]) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def split_list(items: List[str], ratios: Tuple[float, float, float], seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle and split `items` into three lists according to `ratios`.

    Uses deterministic `seed` for reproducibility.
    """
    random.seed(seed)  # fix seed for reproducible splits
    items_copy = items[:]  # copy original list
    random.shuffle(items_copy)  # shuffle in-place

    total = len(items_copy)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    # remaining go to test to ensure sum == total
    n_test = total - n_train - n_val

    train = items_copy[:n_train]
    val = items_copy[n_train:n_train + n_val]
    test = items_copy[n_train + n_val:]
    return train, val, test


def copy_split(items: List[str], src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path) -> int:
    """Copy image and label files for the list of basenames. Returns number copied."""
    count = 0
    for im in items:
        base = Path(im).stem
        src_im = src_images / im
        src_lbl = src_labels / (base + '.txt')
        dst_im = dst_images / im
        dst_lbl = dst_labels / (base + '.txt')
        shutil.copy2(src_im, dst_im)  # copy image preserving metadata
        shutil.copy2(src_lbl, dst_lbl)  # copy label
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='Split labeled images into train/val/test sets (70/15/15).')
    parser.add_argument('--images-dir', type=Path, default=DEFAULT_IMAGES_DIR, help='Source images directory containing selected images.')
    parser.add_argument('--labels-dir', type=Path, default=DEFAULT_LABELS_DIR, help='Source labels directory matching images.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used for shuffling (default: 42).')
    args = parser.parse_args()

    # ensure source directories exist
    if not args.images_dir.is_dir():
        raise SystemExit(f'Images directory not found: {args.images_dir}')
    if not args.labels_dir.is_dir():
        raise SystemExit(f'Labels directory not found: {args.labels_dir}')

    # collect images that have labels
    matched_images = list_images_with_labels(args.images_dir, args.labels_dir)

    total = len(matched_images)  # total eligible images
    print(f'Found {total} images with matching labels. Proceeding to split.')

    # create output directories
    make_dirs([OUT_TRAIN_IMAGES, OUT_VAL_IMAGES, OUT_TEST_IMAGES,
               OUT_TRAIN_LABELS, OUT_VAL_LABELS, OUT_TEST_LABELS])

    # split list deterministically
    train, val, test = split_list(matched_images, RATIOS, seed=args.seed)

    # copy files for each split
    c_train = copy_split(train, args.images_dir, args.labels_dir, OUT_TRAIN_IMAGES, OUT_TRAIN_LABELS)
    c_val = copy_split(val, args.images_dir, args.labels_dir, OUT_VAL_IMAGES, OUT_VAL_LABELS)
    c_test = copy_split(test, args.images_dir, args.labels_dir, OUT_TEST_IMAGES, OUT_TEST_LABELS)

    # print summary counts
    print('\nSplit summary:')
    print(f'  Train: {c_train} images -> {OUT_TRAIN_IMAGES} and {OUT_TRAIN_LABELS}')
    print(f'  Val:   {c_val} images -> {OUT_VAL_IMAGES} and {OUT_VAL_LABELS}')
    print(f'  Test:  {c_test} images -> {OUT_TEST_IMAGES} and {OUT_TEST_LABELS}')


if __name__ == '__main__':
    main()
