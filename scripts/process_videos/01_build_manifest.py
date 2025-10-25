import csv, os, re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE = Path(os.getenv("DATASET_PATH", "ellie_dataset/processed"))
CLASSES = os.getenv("CLASSES", "poop,not_poop,coprophagy").split(",")

# Configuration from environment
MANIFEST_FILENAME = os.getenv("MANIFEST_FILENAME", "manifest.csv")
METADATA_FILENAME = os.getenv("METADATA_FILENAME", "metadata_out.csv")
VIDEO_EXTENSION = os.getenv("VIDEO_EXTENSION", ".mp4")

out_csv = BASE / MANIFEST_FILENAME
rows = []

def dur_from_metadata(fname):
    # tenta ler duration do metadata_out.csv já gerado pelo seu script de ffmpeg
    meta_csv = BASE / METADATA_FILENAME
    if meta_csv.exists():
        with open(meta_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r["filename"] == f"{fname.parent.name}/{fname.name}":
                    try:
                        return float(r["duration_sec"])
                    except:
                        return None
    return None

for label in CLASSES:
    d = BASE / label
    if not d.exists(): 
        continue
    for f in sorted(d.glob(f"*{VIDEO_EXTENSION}")):
        rows.append({
            "path": str(f),
            "label": label,
            "duration_sec": dur_from_metadata(f) or "",
        })

out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path","label","duration_sec"])
    w.writeheader()
    w.writerows(rows)

print(f"✅ manifest salvo em: {out_csv} (total {len(rows)} vídeos)")