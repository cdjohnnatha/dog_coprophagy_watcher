import csv, os, sys
import cv2
import numpy as np
from pathlib import Path
from itertools import product
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
BASE = Path(os.getenv("DATASET_PATH", "ellie_dataset/processed"))
VAL_MANIFEST_FILENAME = os.getenv("VAL_MANIFEST_FILENAME", "manifest_val.csv")
VIDEO_FPS = float(os.getenv("VIDEO_FPS", "8.0"))

# Optical flow parameters
FLOW_WIN = int(os.getenv("FLOW_WIN", "15"))
FLOW_LVL = int(os.getenv("FLOW_LVL", "2"))
FLOW_ITER = int(os.getenv("FLOW_ITER", "3"))
FLOW_LOW_MAG = float(os.getenv("FLOW_LOW_MAG", "0.6"))
FLOW_LOW_RATIO_GATE = float(os.getenv("FLOW_LOW_RATIO_GATE", "0.65"))

VAL_CSV = BASE / VAL_MANIFEST_FILENAME

def video_features(path):
    cap = cv2.VideoCapture(str(path))
    ok, prev = cap.read()
    if not ok:
        return None
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    low_mov_flags = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None,
                                            0.5, FLOW_LVL, FLOW_WIN, FLOW_ITER, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        m = np.median(mag)
        low_mov_flags.append(m < FLOW_LOW_MAG)
        prev = gray
    cap.release()

    if len(low_mov_flags) == 0:
        return {"still_ratio":0.0, "act_duration_s":0.0, "nframes":0}

    still_ratio = sum(low_mov_flags) / len(low_mov_flags)

    # maior sequência contínua de “baixo movimento”
    max_run = 0
    run = 0
    for f in low_mov_flags:
        if f: 
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    # fps alvo do seu dataset processado
    act_duration_s = max_run / VIDEO_FPS
    return {
        "still_ratio": float(still_ratio),
        "act_duration_s": float(act_duration_s),
        "nframes": len(low_mov_flags)
    }

# lê validação
val_rows = []
with open(VAL_CSV, newline="") as f:
    for r in csv.DictReader(f):
        if Path(r["path"]).exists():
            val_rows.append(r)

# extrai features
feats = []
for r in val_rows:
    d = video_features(Path(r["path"]))
    if d is None: continue
    d["label"] = r["label"]
    d["path"] = r["path"]
    feats.append(d)

# grid search simples para thresholds
# usamos still_ratio como proxy de "SQUAT_SCORE_POOP"
# e act_duration_s como proxy de "POOP_MIN_DURATION_S"
best = None
cand_still = np.linspace(0.55, 0.85, 13)   # 0.55 .. 0.85
cand_dur   = np.linspace(3.5, 7.0, 15)     # 3.5s .. 7.0s

def eval_thresh(sr, dur):
    tp=fp=tn=fn=0
    for d in feats:
        pred_poof = (d["still_ratio"] >= sr) and (d["act_duration_s"] >= dur)
        is_poop = (d["label"] == "poop")
        if pred_poof and is_poop: tp += 1
        elif pred_poof and not is_poop: fp += 1
        elif (not pred_poof) and (not is_poop): tn += 1
        else: fn += 1
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return {"prec":prec, "rec":rec, "f1":f1, "tp":tp, "fp":fp, "tn":tn, "fn":fn}

for sr in cand_still:
    for dur in cand_dur:
        m = eval_thresh(sr, dur)
        if (best is None) or (m["f1"] > best["f1"]):
            best = {"sr":sr, "dur":dur, **m}

print("=== Resultado (val set) ===")
print(f"F1={best['f1']:.3f}  Prec={best['prec']:.3f}  Rec={best['rec']:.3f}")
print(f"TP={best['tp']} FP={best['fp']} TN={best['tn']} FN={best['fn']}")
print(f"SUGERIR .env:")
print(f"  SQUAT_SCORE_POOP={best['sr']:.2f}")
print(f"  POOP_MIN_DURATION_S={best['dur']:.1f}")
print("\nDica: se FP alto à noite, suba SQUAT_SCORE_POOP em +0.02; se FN alto, reduza POOP_MIN_DURATION_S em -0.5s.")