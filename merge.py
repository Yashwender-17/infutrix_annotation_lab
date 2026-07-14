"""
merge.py
────────
Final merger script — runs after all annotators have returned their
ocr_training/ folders. Merges validated annotations into one clean dataset.

Usage:
    python merge.py

Steps:
  1. Ask for folder containing all annotator outputs
  2. Scan each subfolder for images/ + annotations.json
  3. Check validation status of every image entry
  4. Log all non-validated images with detail
  5. Merge only validated images into:
       merged_dataset_YYYY-MM-DD/
         images/            <- flat copy of all validated images
         annotations.json   <- [{"filename": "...", "text": "..."}]
  6. Save a full merge_log_DATE.txt
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (Edit these directly)
# ─────────────────────────────────────────────────────────────────────────────
# Put all the completed annotator folders into this directory.
# (e.g., place Raunak_2026-07-11_123045 inside received_annotations/)
SOURCE_FOLDER = 'received_annotations'
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = '') -> str:
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


def _make_output_dir(base_dir: Path) -> Path:
    """Create a date-stamped output directory, appending _v2, _v3 if needed."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    candidate = base_dir / f"merged_dataset_{date_str}"
    if not candidate.exists():
        return candidate
    version = 2
    while True:
        candidate = base_dir / f"merged_dataset_{date_str}_v{version}"
        if not candidate.exists():
            return candidate
        version += 1


def _load_annotations(person_dir: Path) -> dict | None:
    """
    Load annotations.json from a person's ocr_training folder.
    Returns the dict or None if missing/corrupt.
    """
    json_file = person_dir / 'annotations.json'
    if not json_file.exists():
        return None
    try:
        with open(json_file, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠️  Could not read {json_file}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 65)
    print("  🧬  Infutrix Annotation Lab — Final Dataset Merger")
    print("═" * 65 + "\n")

    run_ts = datetime.now()

    # ── 1. Source folder ──────────────────────────────────────────────────────
    src_root = Path(__file__).parent / SOURCE_FOLDER

    if not src_root.exists():
        print(f"❌  Source folder not found: {src_root}")
        print(f"   Please check the SOURCE_FOLDER variable at the top of the script.")
        print(f"   (You need to create a '{SOURCE_FOLDER}' folder and put the completed annotator folders inside it).")
        sys.exit(1)

    # ── 2. Discover person folders ────────────────────────────────────────────
    person_dirs = sorted([
        d for d in src_root.iterdir()
        if d.is_dir() and (d / 'annotations.json').exists()
    ])

    if not person_dirs:
        print(f"\n❌  No annotator subfolders with annotations.json found in:\n    {src_root}")
        sys.exit(1)

    print(f"\n📂  Found {len(person_dirs)} annotator folder(s):\n")

    # ── 3. Check validation status ────────────────────────────────────────────
    # Each person's annotations.json is expected to be a dict:
    # { "filename.jpg": {"ocr_text": "...", "status": "validated"|"pending"|"skipped", ...} }

    person_stats    = []  # list of dicts per person
    skipped_detail  = []  # (person_name, filename, status) for non-validated

    for pdir in person_dirs:
        ann = _load_annotations(pdir)
        if ann is None:
            print(f"  ⚠️  {pdir.name:<30} — annotations.json missing or unreadable, skipping.")
            continue

        total     = len(ann)
        validated = 0
        pending   = 0
        has_warn  = False

        for filename, meta in ann.items():
            status = ''
            if isinstance(meta, dict):
                status = meta.get('status', '').lower()
            # If meta is just a string (bare ocr_text), treat as unvalidated
            if status == 'validated':
                validated += 1
            else:
                pending += 1
                skipped_detail.append((pdir.name, filename, status or 'unknown'))
                has_warn = True

        icon = "⚠️ " if has_warn else "✅"
        print(f"  {icon} {pdir.name:<30} — {total} images, "
              f"{validated} validated, {pending} pending/skipped")

        person_stats.append({
            'dir':       pdir,
            'name':      pdir.name,
            'ann':       ann,
            'total':     total,
            'validated': validated,
            'pending':   pending,
        })

    # ── 4. Warn about non-validated ───────────────────────────────────────────
    if skipped_detail:
        print(f"\n⚠️   WARNING: {len(skipped_detail)} image(s) are NOT validated across all annotators.")
        print(f"\n📋  Non-validated image detail:")
        for (person, fname, status) in skipped_detail[:50]:   # show first 50 inline
            print(f"    [{person}]  {fname}  —  status: {status}")
        if len(skipped_detail) > 50:
            print(f"    ... and {len(skipped_detail) - 50} more (see log file)")

    # ── 5. Confirm merge ──────────────────────────────────────────────────────
    print()
    if skipped_detail:
        print("⚠️   Proceeding to merge ONLY the validated images (skipping the rest).")
    else:
        print("✅  All images validated — proceeding with full merge.")

    # ── 6. Merge ──────────────────────────────────────────────────────────────
    output_dir = _make_output_dir(Path(__file__).parent)
    out_img_dir = output_dir / 'images'
    out_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦  Merging validated images...")

    merged_entries = []   # [{"filename": "...", "text": "..."}]
    merged_count   = 0
    skip_count     = 0
    conflict_count = 0
    seen_filenames: set[str] = set()

    for ps in person_stats:
        pdir = ps['dir']
        ann  = ps['ann']
        img_dir = pdir / 'images'

        for filename, meta in ann.items():
            status = ''
            ocr_text = ''

            if isinstance(meta, dict):
                status   = meta.get('status', '').lower()
                ocr_text = meta.get('ocr_text', '')
            elif isinstance(meta, str):
                # Legacy bare string format
                ocr_text = meta
                status   = ''

            if status != 'validated':
                skip_count += 1
                continue

            # Source image
            src_img = img_dir / filename
            if not src_img.exists():
                print(f"  ⚠️  Image not found on disk: {src_img} — skipping.")
                skip_count += 1
                continue

            # Handle filename conflicts
            dest_filename = filename
            if filename in seen_filenames:
                stem   = src_img.stem
                suffix = src_img.suffix
                dest_filename = f"{stem}_{ps['name']}{suffix}"
                conflict_count += 1
                print(f"  ⚠️  Filename conflict: {filename} → renamed to {dest_filename}")

            seen_filenames.add(dest_filename)

            # Copy image
            shutil.copy2(src_img, out_img_dir / dest_filename)

            # Add to merged list
            merged_entries.append({
                'filename': dest_filename,
                'text':     ocr_text,
            })
            merged_count += 1

    # ── 7. Write final annotations.json ──────────────────────────────────────
    final_json_path = output_dir / 'annotations.json'
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_entries, f, indent=2, ensure_ascii=False)

    # ── 8. Write merge log ────────────────────────────────────────────────────
    date_str  = run_ts.strftime("%Y-%m-%d")
    log_path  = Path(__file__).parent / f"merge_log_{date_str}.txt"
    # Append if already exists (e.g. v2 run same day)
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n{'='*65}\n")
        log.write(f"Merge Run: {run_ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Source   : {src_root}\n")
        log.write(f"Output   : {output_dir}\n")
        log.write(f"{'='*65}\n\n")

        log.write("SOURCE FOLDERS:\n")
        for ps in person_stats:
            log.write(f"  {ps['name']:<30} : {ps['total']} images | "
                      f"{ps['validated']} validated | {ps['pending']} pending/skipped\n")

        if skipped_detail:
            log.write(f"\nSKIPPED (not validated) — {len(skipped_detail)} total:\n")
            for (person, fname, status) in skipped_detail:
                log.write(f"  [{person}]  {fname}  status={status}\n")
        else:
            log.write("\nNo skipped images — all were validated.\n")

        log.write(f"\nMERGED : {merged_count} images\n")
        log.write(f"SKIPPED: {skip_count} images\n")
        if conflict_count:
            log.write(f"RENAMED: {conflict_count} images (filename conflicts)\n")
        log.write(f"OUTPUT : {output_dir}\n")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    print(f"\n✅  Merged:  {merged_count} images")
    print(f"⏭️   Skipped: {skip_count} images (not validated)")
    if conflict_count:
        print(f"🔀  Renamed: {conflict_count} images (filename conflicts resolved)")
    print(f"\n📁  Output folder : {output_dir}")
    print(f"    ├── images/          ({merged_count} images)")
    print(f"    └── annotations.json ({merged_count} entries)")
    print(f"\n📝  Merge log saved : {log_path}")
    print("\n🎉  Done!\n")


if __name__ == '__main__':
    main()
