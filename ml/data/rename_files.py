#!/usr/bin/env python3
"""
Simple folder renamer: make all files follow the pattern 'prefix_###.ext'.

Rules:
- Work in a single folder (non-recursive)
- Keep already-correct files as-is (e.g., 'prefix_001.ext' is kept)
- Do not overwrite: if a target exists, skip that number and use the next
- Hidden files (like .DS_Store) are ignored

Usage:
  python rename_files.py <folder_path> <prefix>

Example:
  python rename_files.py ./videos coprophagy
  SomeRandomName.mp4 -> coprophagy_001.mp4
"""

import argparse
import os
import re
from typing import Dict, List, Set, Tuple


PREFIXED_RE_CACHE: Dict[str, re.Pattern] = {}


def _is_hidden(filename: str) -> bool:
    return filename.startswith(".")


def _prefixed_regex(prefix: str) -> re.Pattern:
    # Cache per prefix to avoid recompilation
    if prefix not in PREFIXED_RE_CACHE:
        PREFIXED_RE_CACHE[prefix] = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    return PREFIXED_RE_CACHE[prefix]


def _list_files(folder_path: str) -> List[str]:
    return [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and not _is_hidden(f)
    ]


def rename_files(folder_path: str, prefix: str) -> None:
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return

    files = _list_files(folder_path)
    if not files:
        print(f"No files found in '{folder_path}'.")
        return

    files.sort()  # deterministic order

    # Track occupied indices per extension by scanning existing correctly named files
    occupied_by_ext: Dict[str, Set[int]] = {}
    already_correct: Set[str] = set()
    rx = _prefixed_regex(prefix)

    for name in files:
        stem, ext = os.path.splitext(name)
        if rx.match(stem):
            try:
                idx = int(rx.match(stem).group(1))
            except Exception:
                idx = None  # should not happen
            if idx is not None:
                if ext not in occupied_by_ext:
                    occupied_by_ext[ext] = set()
                occupied_by_ext[ext].add(idx)
                already_correct.add(name)

    # Plan renames for files that are not already correct
    plan: List[Tuple[str, str]] = []
    PAD = 3
    for name in files:
        if name in already_correct:
            continue
        stem, ext = os.path.splitext(name)
        used = occupied_by_ext.setdefault(ext, set())

        idx = 1
        while idx in used:
            idx += 1

        target = f"{prefix}_{idx:0{PAD}d}{ext}"
        # Reserve the index immediately to avoid duplicates within this run
        used.add(idx)
        plan.append((name, target))

    # Execute plan without overwriting; targets are guaranteed unique and unused
    for src, dst in plan:
        src_path = os.path.join(folder_path, src)
        dst_path = os.path.join(folder_path, dst)
        if os.path.exists(dst_path):
            # Should not happen due to reservation, but double-safeguard
            print(f"Skip (exists): {dst}")
            continue
        os.rename(src_path, dst_path)
        print(f"Renamed: {src} -> {dst}")

    print(f"\nCompleted. {len(plan)} files processed. {len(already_correct)} already matched.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename all files in a folder to 'prefix_###.ext' without overwriting."
    )
    parser.add_argument("folder_path", help="Path to the folder containing files")
    parser.add_argument("prefix", help="Prefix to use for renamed files")
    args = parser.parse_args()

    rename_files(args.folder_path, args.prefix)


if __name__ == "__main__":
    main()
