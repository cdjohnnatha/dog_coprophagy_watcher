import glob
import os
import random
from typing import List, Sequence, Tuple

import numpy as np


Item = Tuple[str, float, float]


def index_dataset(root_dir: str) -> List[Item]:
    exts = ("*.mp4", "*.mov", "*.mkv", "*.avi")
    folders = {
        "Poop": os.path.join(root_dir, "Poop"),
        "Not_Poop": os.path.join(root_dir, "Not_Poop"),
        "Coprophagy": os.path.join(root_dir, "Coprophagy"),
        "Not_Coprophagy": os.path.join(root_dir, "Not_Coprophagy"),
    }
    for k, p in folders.items():
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Pasta não encontrada: {p}")

    def glob_many(folder: str) -> Sequence[str]:
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, e)))
        return sorted(files)

    items: List[Item] = []
    for f in glob_many(folders["Poop"]):
        items.append((f, 1, np.nan))
    for f in glob_many(folders["Not_Poop"]):
        items.append((f, 0, np.nan))

    for f in glob_many(folders["Coprophagy"]):
        items.append((f, np.nan, 1))
    for f in glob_many(folders["Not_Coprophagy"]):
        items.append((f, np.nan, 0))

    seen, uniq = set(), []
    for path, y_po, y_co in items:
        if path in seen:
            continue
        seen.add(path)
        uniq.append((path, y_po, y_co))
    return uniq


def train_val_split(items: List[Item], val_ratio: float = 0.20, seed: int = 42):
    random.Random(seed).shuffle(items)
    n = len(items)
    val_n = max(1, int(n * val_ratio))
    return items[val_n:], items[:val_n]


