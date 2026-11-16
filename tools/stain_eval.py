"""Offline evaluation (T141, 2026-11-14): LAB colour stain measure vs the grey new_stain_area on stored visits.

Run: cd /local_apps/ellie && nice -n 19 .venv/bin/python -I -W ignore tools/stain_eval.py <output folder>
"""
import io, json, math, pathlib, sys
sys.path.insert(0, "/local_apps/ellie/src")
from PIL import Image, ImageCms
from ellie import watch as w

W = 192
SRGB, LABP = ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB")
TR = ImageCms.buildTransformFromOpenProfiles(SRGB, LABP, "RGB", "LAB")

def lab(jpeg):
    im = Image.open(io.BytesIO(jpeg)); im.draft("RGB", (W * 2, W * 2)); im = im.convert("RGB")
    h = round(W * im.height / im.width); im = im.resize((W, h))
    L, a, b = (list(c.getdata()) for c in ImageCms.applyTransform(im, TR).split())
    return [v * 100 / 255 for v in L], a, b, h

def regions(h):
    x1, y1, x2, y2 = w.PAD_BOX; m = w.CONTROL_MARGIN; pad, floor = [], []
    for r in range(h):
        for c in range(W):
            fx, fy = (c + .5) / W, (r + .5) / h
            if x1 <= fx <= x2 and y1 <= fy <= y2: pad.append(r * W + c)
            elif not (x1 - m <= fx <= x2 + m and y1 - m <= fy <= y2 + m): floor.append(r * W + c)
    return pad, floor

def median(v):
    s = sorted(v); return s[len(s) // 2]

def lab_metric(bj, aj, thresholds, min_blob=15):
    Lb, ab, bb, h = lab(bj); La, aa, ba, h2 = lab(aj)
    if h != h2: return None
    pad, floor = regions(h)
    dL = [La[i] - Lb[i] for i in range(len(La))]; da = [aa[i] - ab[i] for i in range(len(La))]; db = [ba[i] - bb[i] for i in range(len(La))]
    mL, ma, mb = median([dL[i] for i in floor]), median([da[i] for i in floor]), median([db[i] for i in floor])
    sat = sum(abs(ab[i] - 128) + abs(bb[i] - 128) for i in floor) / len(floor)  # ~0 for IR
    padset = set(pad); out = {}
    for t in thresholds:
        def stained(i):
            x, y, z = dL[i] - mL, da[i] - ma, db[i] - mb
            return math.sqrt(x * x + y * y + z * z) >= t and (x < 0 or z > 0)
        hit = {i for i in pad if stained(i)}
        seen, kept = set(), 0
        for i in hit:  # 4-neighbour components, drop specks
            if i in seen: continue
            stack, comp = [i], 0; seen.add(i)
            while stack:
                j = stack.pop(); comp += 1
                r, c = divmod(j, W)
                for k in ((j - 1) if c else -1, (j + 1) if c < W - 1 else -1, j - W, j + W):
                    if k in hit and k not in seen: seen.add(k); stack.append(k)
            if comp >= min_blob: kept += comp
        fl = sum(1 for i in floor if stained(i)) / len(floor)
        out[t] = (kept / len(pad), fl)
    return out, sat

labels = {}
for line in open("/media/ellie/dataset/labels.jsonl"):
    d = json.loads(line); labels[d["visit_id"]] = d
truth_fix = {"v_20261113_235855": "pee"}
exclude = {"v_20261107_095424": "owner said mixed up", "v_20261109_125318": "pee on the floor, pad unchanged"}
ambiguous = {"v_20261112_205918"}
missed = {"v_20261113_235855", "v_20261110_201444", "v_20261108_002556", "v_20261114_093710"}  # Gemini said no_pee
after_override = {"v_20261114_093710": "after_person.jpg"}
TH = [6, 8, 10, 12, 15, 19, 25]
rows = []
for vd in sorted(pathlib.Path("/media/ellie/visits").glob("*/*/*/v_*")):
    vid = vd.name; lab_ = labels.get(vid)
    truth = truth_fix.get(vid, lab_["label"] if lab_ else None)
    if truth is None or vid in exclude: continue
    if lab_ and lab_.get("poop_observed", "none") != "none": continue
    b = vd / "before.jpg"; a = vd / after_override.get(vid, "after.jpg")
    if not (b.exists() and a.exists()): continue
    frigate = (vd / "during.jpg").exists() or (vd / "during_own.jpg").exists()
    bj, aj = b.read_bytes(), a.read_bytes()
    grey = w.new_stain_area(bj, aj)
    res = lab_metric(bj, aj, TH)
    if res is None or grey is None: continue
    m, sat = res
    rows.append(dict(id=vid, truth=truth, frigate=frigate, grey=grey[0], gfloor=grey[1], lab=m, ir=sat < 3))
json.dump(rows, open(pathlib.Path(sys.argv[1]) / "rows.json", "w"))

def summary(name, val, floor_ok):
    pee = [r for r in rows if r["truth"] == "pee" and r["frigate"] and r["id"] not in ambiguous]
    nop = [r for r in rows if r["truth"] == "no_pee" and r["frigate"] and r["id"] not in ambiguous and floor_ok(r)]
    ms = sorted((val(r), r["id"][2:]) for r in pee if r["id"] in missed)
    lo = min(v for v, _ in ms)
    fp = sorted(((val(r), r["id"][2:]) for r in nop), reverse=True)
    above = [x for x in fp if x[0] >= lo]
    pees = sorted(val(r) for r in pee)
    print(f"{name:10} missed pees: " + " ".join(f"{v*100:.1f}%({i[4:8]}_{i[9:13]})" for v, i in ms)
          + f" | all pees n={len(pees)} min {pees[0]*100:.1f}% med {pees[len(pees)//2]*100:.1f}%"
          + f" | no_pee n={len(nop)} top: " + " ".join(f"{v*100:.1f}%" for v, _ in fp[:5])
          + f" | no_pee >= weakest missed: {len(above)}")

summary("grey", lambda r: r["grey"], lambda r: r["gfloor"] <= 0.06)
for t in TH:
    summary(f"LAB dE{t}", lambda r, t=t: r["lab"][str(t)][0] if isinstance(next(iter(r["lab"])), str) else r["lab"][t][0], lambda r, t=t: r["lab"][t][1] <= 0.05)
amb = [r for r in rows if r["id"] in ambiguous]
for r in amb: print("ambiguous", r["id"], "grey", round(r["grey"]*100, 1), "LAB", {t: round(r["lab"][t][0]*100, 1) for t in TH})
print("IR visits:", sum(r["ir"] for r in rows), "of", len(rows))
