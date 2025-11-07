# scripts/snippets/eval_val.py
import os
import argparse
import numpy as np
import tensorflow as tf

from ml.data.dataset import make_dataset
from ml.data.data_index import index_dataset, train_val_split
from ml.models.model import build_model

DATA_DIR = "assets/normalized"
FRAMES   = 16
SIZE     = 224
BATCH    = 4
VAL_SPLIT = 0.20

DEFAULT_OUTDIR = "datasets/ellie_multihead"

def collect_from_dataset(ds):
    """Extrai X, y_poop, y_copro do tf.data suportando rótulos como tupla ou dict."""
    Xs, Yp, Yc = [], [], []
    for batch in ds.as_numpy_iterator():
        # batch can be (x, ydict) or (x, (y1,y2)) or (x, y1, y2)
        if isinstance(batch, tuple) and len(batch) == 2:
            x, y = batch
            if isinstance(y, dict):
                y1 = y.get('poop', None)
                if y1 is None:
                    y1 = y.get(b'poop')
                y2 = y.get('copro', None)
                if y2 is None:
                    y2 = y.get(b'copro')
            else:
                y1, y2 = y  # assume tuple/list of two
        elif isinstance(batch, tuple) and len(batch) == 3:
            x, y1, y2 = batch
        else:
            raise RuntimeError("Formato inesperado de batch no dataset de validação.")

        Xs.append(x)
        Yp.append(y1)
        Yc.append(y2)
    X  = np.concatenate(Xs, axis=0)
    yp = np.concatenate(Yp, axis=0).reshape(-1, 1)
    yc = np.concatenate(Yc, axis=0).reshape(-1, 1)
    return X, yp, yc

def bin_metrics(y_true, y_prob, thr=0.5):
    y_true = y_true.astype(int).reshape(-1)
    y_pred = (y_prob.reshape(-1) >= thr).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    tot = max(1, tp + tn + fp + fn)
    acc  = (tp + tn) / tot
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return tp, fp, tn, fn, acc, prec, rec, f1

def best_f1(y_true, y_prob):
    best = (0.0, 0.50)
    for thr in np.linspace(0.0, 0.99, 100):
        _, _, _, _, _, _, _, f1 = bin_metrics(y_true, y_prob, thr)
        if f1 > best[0]:
            best = (f1, float(thr))
    return best

def prob_stats(name: str, y_prob: np.ndarray, mask: np.ndarray):
    vals = y_prob[mask].reshape(-1)
    if vals.size == 0:
        print(f"{name} probs: (no valid samples)")
        return
    q = np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"{name} probs: min={q[0]:.3f} p25={q[1]:.3f} p50={q[2]:.3f} p75={q[3]:.3f} max={q[4]:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.environ.get("ELLIE_OUTDIR", DEFAULT_OUTDIR))
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--val_split", type=float, default=VAL_SPLIT)
    args = ap.parse_args()

    outdir = args.outdir
    data_dir = args.data_dir
    frames = args.frames
    size = args.size
    batch = args.batch
    val_split = args.val_split

    # isola cache do eval para este outdir
    os.environ["ELLIE_OUTDIR"] = outdir
    # monta índice de validação
    items = index_dataset(data_dir)
    _, val_items = train_val_split(items, val_ratio=val_split)

    # carrega modelo/weights do treino
    ckpt_model = os.path.join(outdir, "ckpt_best.keras")
    ckpt_weights = os.path.join(outdir, "ckpt_best.weights.h5")
    saved = os.path.join(outdir, "saved_model")
    model = None
    if os.path.isfile(ckpt_weights):
        # Pesos apenas
        model = build_model(frames, size, use_metrics=False)
        model.load_weights(ckpt_weights, by_name=True, skip_mismatch=True)
    elif os.path.isfile(ckpt_model):
        # Modelo completo salvo em .keras
        model = tf.keras.models.load_model(ckpt_model, compile=False)
    elif os.path.isdir(saved):
        model = tf.keras.models.load_model(saved, compile=False)
    else:
        raise FileNotFoundError(f"Nem checkpoint nem SavedModel encontrados em {outdir}")

    # Ajusta frames/size do dataset conforme input do modelo salvo
    try:
        ishape = model.input_shape
        if isinstance(ishape, list):
            ishape = ishape[0]
        req_frames = int(ishape[1])
        req_size = int(ishape[2])
        if frames != req_frames or size != req_size:
            frames = req_frames
            size = req_size
    except Exception:
        pass

    # cria dataset de validação com as dimensões esperadas pelo modelo
    val_ds = make_dataset(val_items, frames, size, batch, shuffle=False, parallel_reads=None, debug_io=False)

    # coleta tensores e rótulos NA MESMA ORDEM do ds
    X, y_po, y_co = collect_from_dataset(val_ds)
    print(f">> Val samples (dataset): {len(X)}")

    # predições alinhadas (evita exaustão do dataset usando o tensor diretamente)
    preds = model.predict(X, batch_size=batch, verbose=0)
    p_po = preds["poop"].reshape(-1, 1)
    p_co = preds["copro"].reshape(-1, 1)

    # aplica máscara de válidos por cabeça (y != -1)
    m_po = (y_po != -1).reshape(-1)
    m_co = (y_co != -1).reshape(-1)

    # POOP
    if m_po.any():
        y_true_po = y_po[m_po].astype(int)
        y_prob_po = p_po[m_po]
        prob_stats("POOP", p_po, m_po)
        tp, fp, tn, fn, acc, prec, rec, f1 = bin_metrics(y_true_po, y_prob_po, thr=0.5)
        bf1, bthr = best_f1(y_true_po, y_prob_po)
        print("\n== POOP ==")
        print(f"válidos={m_po.sum()}  pos={int((y_true_po==1).sum())}  neg={int((y_true_po==0).sum())}")
        print(f"Threshold: 0.50")
        print(f"Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1-score:  {f1:.3f}")
        print(f"POOP: melhor F1={bf1:.3f} em threshold={bthr:.2f}")
    else:
        print("\n== POOP ==\nSem amostras válidas no VAL.")

    # COPRO
    if m_co.any():
        y_true_co = y_co[m_co].astype(int)
        y_prob_co = p_co[m_co]
        prob_stats("COPRO", p_co, m_co)
        tp, fp, tn, fn, acc, prec, rec, f1 = bin_metrics(y_true_co, y_prob_co, thr=0.5)
        bf1_co, bthr_co = best_f1(y_true_co, y_prob_co)
        print("\n== COPRO ==")
        print(f"válidos={m_co.sum()}  pos={int((y_true_co==1).sum())}  neg={int((y_true_co==0).sum())}")
        print(f"Threshold: 0.50")
        print(f"Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1-score:  {f1:.3f}")
        print(f"COPRO: melhor F1={bf1_co:.3f} em threshold={bthr_co:.2f}")
    else:
        print("\n== COPRO ==\nSem amostras válidas no VAL.")

    # Persist best thresholds for deployment
    try:
        import json, time
        out = {
            "frames": int(frames),
            "size": int(size),
            "timestamp": int(time.time()),
        }
        if m_po.any():
            out["poop_threshold"] = float(bthr)
            out["poop_best_f1"] = float(bf1)
        if m_co.any():
            out["copro_threshold"] = float(bthr_co)
            out["copro_best_f1"] = float(bf1_co)
        with open(os.path.join(outdir, "thresholds.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved thresholds.json to {outdir}")
    except Exception as e:
        print(f"(warn) failed to write thresholds.json: {e}")

if __name__ == "__main__":
    main()