"""
distribute.py
─────────────
Standalone CLI script for distributing a YOLO-format dataset across N people.

Usage:
    python distribute.py

What it does:
  1. Scans train/ test/ valid/ subfolders for all images + matching labels
  2. Asks how many people and their names
  3. Splits images (equal or weighted) and copies into per-person folders:
       distributions/person_<Name>/images/
       distributions/person_<Name>/labels/
       distributions/person_<Name>/assignment.json
"""

import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (Edit these directly)
# ─────────────────────────────────────────────────────────────────────────────
# Set the name of your dataset folder here
DATASET_FOLDER = 'yolo_format_data'

# Add the names of the people you want to distribute to
ANNOTATOR_NAMES = [
    'Raunak',
    'Akash',
    'John',
    'Sarah'
]

# Distribution strategy:
# 1 = Equal split (everyone gets roughly the same amount)
# 3 = By source prefix (groups images with the same prefix together)
STRATEGY = '1'
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = '') -> str:
    """Print prompt, read input, return stripped response."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else default


def _confirm(prompt: str) -> bool:
    ans = _ask(prompt + " [Y/n]", "Y").upper()
    return ans in ("Y", "YES", "")


def _collect_images(dataset_path: Path) -> list[dict]:
    """
    Walk train/, test/, valid/ image folders and return list of:
      {'filename': str, 'image_path': Path, 'label_path': Path | None, 'split': str}
    """
    splits = ['train', 'test', 'valid']
    items  = []

    for split in splits:
        img_dir   = dataset_path / split / 'images'
        label_dir = dataset_path / split / 'labels'
        if not img_dir.exists():
            continue
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            label_file = label_dir / (img_file.stem + '.txt')
            items.append({
                'filename':   img_file.name,
                'image_path': img_file,
                'label_path': label_file if label_file.exists() else None,
                'split':      split,
            })

    return items


def _split_equally(items: list, n: int) -> list[list]:
    """Divide items into n groups as evenly as possible."""
    groups = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        groups[idx % n].append(item)
    return groups


def _split_weighted(items: list, weights: list[float]) -> list[list]:
    """Divide items according to percentage weights (must sum ~100)."""
    total      = len(items)
    groups     = []
    start      = 0
    n          = len(weights)
    w_sum      = sum(weights)

    for i, w in enumerate(weights):
        if i == n - 1:
            groups.append(items[start:])           # last person gets remainder
        else:
            count = math.floor((w / w_sum) * total)
            groups.append(items[start:start + count])
            start += count

    return groups


def _write_person_folder(out_root: Path, name: str, items: list) -> Path:
    """
    Copy images + labels into out_root/<Name>_<YYYY-MM-DD>/
    Returns the created folder path.
    """
    safe_name = name.strip().replace(' ', '_')
    # Use Year-Month-Day_HourMinuteSecond to prevent any collisions on the same day
    date_str  = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    person_dir = out_root / f"{safe_name}_{date_str}"
    img_dir    = person_dir / 'images'
    lbl_dir    = person_dir / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        # Copy image
        shutil.copy2(item['image_path'], img_dir / item['filename'])
        # Copy label (if exists)
        if item['label_path']:
            shutil.copy2(item['label_path'], lbl_dir / (item['image_path'].stem + '.txt'))

    # Write assignment.json
    assignment = {
        'annotator':    name.strip(),
        'created_at':   datetime.now().isoformat(timespec='seconds'),
        'total_images': len(items),
        'images': [
            {
                'filename':       item['filename'],
                'original_split': item['split'],
            }
            for item in items
        ]
    }
    with open(person_dir / 'assignment.json', 'w', encoding='utf-8') as f:
        json.dump(assignment, f, indent=2, ensure_ascii=False)

    return person_dir


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  🧬  Infutrix Annotation Lab — Dataset Distributor")
    print("═" * 60 + "\n")

    # ── 1. Dataset path ──────────────────────────────────────────────────────
    dataset_path = Path(__file__).parent / DATASET_FOLDER

    if not dataset_path.exists():
        print(f"❌  Dataset path not found: {dataset_path}")
        print(f"   Please check the DATASET_FOLDER variable at the top of the script.")
        sys.exit(1)

    print(f"🔍  Scanning {dataset_path} ...")
    items = _collect_images(dataset_path)

    if not items:
        print("❌  No images found under train/test/valid subfolders.")
        sys.exit(1)

    splits_count = {}
    for it in items:
        splits_count[it['split']] = splits_count.get(it['split'], 0) + 1

    print(f"📊  Found {len(items)} images:")
    for split, cnt in sorted(splits_count.items()):
        print(f"    {split:<8}: {cnt} images")

    # ── 2. Names & People ─────────────────────────────────────────────────────
    names = [n.strip() for n in ANNOTATOR_NAMES if n.strip()]
    n_people = len(names)
    
    if n_people == 0:
        print("❌  No annotator names provided. Please edit ANNOTATOR_NAMES at the top of the script.")
        sys.exit(1)
        
    print(f"\n👥  Distributing to {n_people} people: {', '.join(names)}")

    # ── 3. Strategy ───────────────────────────────────────────────────────────
    print(f"\n📐  Using Strategy [{STRATEGY}]")
    strategy = STRATEGY

    if strategy == '2':
        print(f"\n   Enter weight % for each person (must sum to 100).")
        weights = []
        for nm in names:
            while True:
                try:
                    w = float(_ask(f"   Weight for {nm}"))
                    weights.append(w)
                    break
                except ValueError:
                    print("   Please enter a number.")
        groups = _split_weighted(items, weights)

    elif strategy == '3':
        # Group by the prefix before the first underscore (or full name if no underscore)
        prefix_map: dict[str, list] = {}
        for item in items:
            prefix = item['filename'].split('_')[0] if '_' in item['filename'] else item['filename']
            prefix_map.setdefault(prefix, []).append(item)

        prefixes = sorted(prefix_map.keys())
        print(f"   Found {len(prefixes)} source prefix(es): {', '.join(prefixes)}")
        print(f"   Assigning round-robin to {n_people} people.")
        # Flatten groups by prefix in round-robin order
        flat_by_prefix = []
        for p in prefixes:
            flat_by_prefix.extend(prefix_map[p])
        groups = _split_equally(flat_by_prefix, n_people)

    else:  # default: equal
        groups = _split_equally(items, n_people)

    # ── 5. Preview ────────────────────────────────────────────────────────────
    print("\n📋  Preview:")
    for name, grp in zip(names, groups):
        print(f"    {name:<20}: {len(grp)} images")
    print(f"    {'Total':<20}: {sum(len(g) for g in groups)}")

    if not _confirm("\n✅  Confirm and create folders?"):
        print("Aborted.")
        sys.exit(0)

    # ── 6. Write folders ──────────────────────────────────────────────────────
    out_root = Path(__file__).parent / 'distributions'
    out_root.mkdir(exist_ok=True)

    print()
    for name, grp in zip(names, groups):
        person_dir = _write_person_folder(out_root, name, grp)
        print(f"  ✅  Created: {person_dir}  ({len(grp)} images + {sum(1 for i in grp if i['label_path'])} labels)")

    print(f"\n🎉  Done! Folders are in: {out_root}")
    print("   Transfer each <Name>_<Date> folder via pendrive or any medium.")
    print("   The annotator should paste that folder inside their 'assigned_data' folder.")
    print("   Then they run: python app.py\n")


if __name__ == '__main__':
    main()
