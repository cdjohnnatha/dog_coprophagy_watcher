import csv, random, os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.6"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.2"))
TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.2"))
MANIFEST_FILENAME = os.getenv("MANIFEST_FILENAME", "manifest.csv")

random.seed(RANDOM_SEED)
BASE = Path(os.getenv("DATASET_PATH", "ellie_dataset/processed"))
manifest_csv = BASE / MANIFEST_FILENAME

# proporções sugeridas
SPLIT = {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT}

# lê manifest
rows = []
with open(manifest_csv, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

by_label = defaultdict(list)
for r in rows:
    by_label[r["label"]].append(r)

splits = {"train": [], "val": [], "test": []}

for label, items in by_label.items():
    items = items[:]  # copy
    random.shuffle(items)
    n = len(items)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])
    train = items[:n_train]
    val = items[n_train:n_train+n_val]
    test = items[n_train+n_val:]
    splits["train"].extend(train)
    splits["val"].extend(val)
    splits["test"].extend(test)

for split in ["train","val","test"]:
    out = BASE / f"manifest_{split}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path","label","duration_sec"])
        w.writeheader()
        w.writerows(splits[split])
    print(f"✅ {split}: {len(splits[split])} amostras ({out})")