import os, time, json, threading, io
from collections import deque
from dataclasses import dataclass
import requests
import numpy as np
import cv2
import paho.mqtt.client as mqtt

# ---------- ENV ----------
MQTT_HOST   = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER   = os.getenv("MQTT_USER", "")
MQTT_PASS   = os.getenv("MQTT_PASS", "")
MQTT_PREFIX = os.getenv("MQTT_PREFIX", "frigate")

FR_URL      = os.getenv("FRIGATE_BASE_URL", "http://frigate:5000")
CAM         = os.getenv("CAMERA_NAME", "dog")
ZONE        = os.getenv("TOILET_ZONE", "poop_zone")

SQUAT_THR   = float(os.getenv("SQUAT_SCORE_THRESH", "0.65"))
SQUAT_MIN_S = float(os.getenv("SQUAT_MIN_DURATION_S", "5.0"))
RES_WIN_S   = float(os.getenv("RESIDUE_CONFIRM_WINDOW_S", "20"))
RES_MIN_A   = int(os.getenv("RESIDUE_MIN_AREA", "140"))
RES_STATIC_S= float(os.getenv("RESIDUE_STATIC_SEC", "2.0"))
SNAP_FPS    = float(os.getenv("SNAPSHOT_FPS", "4"))
CHECK_INT_S = float(os.getenv("CHECK_INTERVAL_S", "15"))

ENABLE_DEBUG_WATCHER = bool(os.getenv("ENABLE_DEBUG_WATCHER", "false").lower() in ("1", "true", "on", "yes"))

TOPIC_EVENTS = f"{MQTT_PREFIX}/events"
TOPIC_STATE  = "home/ellie/state"
TOPIC_POOP   = "home/ellie/poop_present"
TOPIC_POOP_EVENT = "home/ellie/poop_event"
FR_PUBLIC   = os.getenv("FR_PUBLIC", os.getenv("FRIGATE_PUBLIC_URL", "http://frigate:5000"))

# ---------- DEBUG / LOGGER ----------
def log(*args, mqtt_debug=True):
    if not ENABLE_DEBUG_WATCHER:
        return
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    if mqtt_debug:
        try:
            mqtt_publish(client, "home/ellie/debug/log", msg)
        except Exception:
            pass

# ---------- HELPERS ----------
def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def mqtt_publish(client, topic, payload):
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, separators=(",", ":"))
    client.publish(topic, payload, qos=0, retain=False)

def safe_roi(img, roi):
    """Recorta ROI dentro da imagem; retorna None se inválida/vazia."""
    if img is None or roi is None:
        return None
    x, y, w, h = roi
    h_img, w_img = img.shape[:2]
    x = max(0, int(x)); y = max(0, int(y))
    x2 = min(x + int(w), w_img); y2 = min(y + int(h), h_img)
    if x >= x2 or y >= y2:
        return None
    return img[y:y2, x:x2]

def frigate_create_event(camera: str, label: str, sub_label: str | None = None) -> str | None:
    try:
        url = f"{FR_URL}/api/events"
        payload = {"camera": camera, "label": label}
        if sub_label:
            payload["sub_label"] = sub_label
        r = requests.post(url, json=payload, timeout=3.0)
        r.raise_for_status()
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        return data.get("id") or data.get("event_id")
    except Exception as e:
        log(f"[frigate] error creating event: {e}")
        return None

def frigate_event_urls(event_id: str) -> tuple[str, str]:
    clip  = f"{FR_URL}/api/events/{event_id}/clip.mp4"
    thumb = f"{FR_URL}/api/events/{event_id}/thumbnail.jpg"
    return clip, thumb

def fetch_snapshot():
    url = f"{FR_URL}/api/{CAM}/latest.jpg"
    r = requests.get(url, timeout=2.0)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def find_recent_poop_event_id(ts_confirm: float) -> str | None:
    try:
        after = int(ts_confirm - 30)
        before = int(ts_confirm + 360)
        url = (f"{FR_URL}/api/events"
               f"?camera={CAM}&label=poop&has_clip=1"
               f"&after={after}&before={before}&limit=5")
        r = requests.get(url, timeout=3.0)
        r.raise_for_status()
        items = r.json() if isinstance(r.json(), list) else []
        if not items:
            return None
        items.sort(key=lambda e: abs(ts_confirm - float(e.get("start_time", 0))))
        return items[0].get("id")
    except Exception:
        return None


# ---------- STATE ----------
@dataclass
class TrackState:
    last_bbox: tuple | None = None
    last_ts: float = 0.0
    squat_start: float = 0.0
    in_squat: bool = False
    bg_roi: np.ndarray | None = None          # baseline rápida durante a postura (opcional)
    roi_anchor: tuple | None = None
    poop_present: bool = False
    poop_roi: tuple | None = None
    poop_centroid: tuple | None = None
    residue_bg: np.ndarray | None = None      # baseline capturada DENTRO da janela de confirmação

state = TrackState()

# ---------- MQTT ----------
def on_connect(c, userdata, flags, rc, properties=None):
    log("🔌 MQTT connected", rc)
    c.subscribe(TOPIC_EVENTS, qos=0)
    mqtt_publish(c, "home/ellie/health", {"ok": True, "ts": now_iso(), "camera": CAM, "zone": ZONE})

def on_message(client, _userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return

    after = data.get("after", {}) or {}
    if after.get("camera") != CAM or after.get("label") != "dog":
        return

    zones = after.get("entered_zones") or after.get("current_zones") or after.get("zones") or []
    if ZONE and ZONE not in zones:
        return

    # bbox
    box = None
    if "box" in after and isinstance(after["box"], list) and len(after["box"]) == 4:
        x, y, w, h = after["box"]
        box = (int(x), int(y), int(x + w), int(y + h))
    elif all(k in after for k in ("top", "left", "bottom", "right")):
        box = (int(after["left"]), int(after["top"]), int(after["right"]), int(after["bottom"]))
    if not box:
        return

    # score
    ratio = float(after.get("ratio", 1.0))
    stationary = bool(after.get("stationary", False))
    speed = float(after.get("current_estimated_speed", 0.0))
    motionless = int(after.get("motionless_count", 0))
    ratio_term = max(0.0, min(1.0, (0.70 - ratio) / 0.20))
    stationary_term = 1.0 if stationary or motionless >= 1 else 0.0
    speed_term = max(0.0, min(1.0, 1.0 - min(speed, 1.0)))
    score = 0.55 * ratio_term + 0.30 * stationary_term + 0.15 * speed_term

    log(f"🐶 Zones={zones}, ratio={ratio:.2f}, stationary={stationary}, speed={speed:.2f}, score={score:.3f}")

    state.last_ts = time.time()
    state.last_bbox = box
    t = time.time()

    if score >= SQUAT_THR:
        try:
            img = fetch_snapshot()
        except Exception as e:
            log(f"⚠️ snapshot fail: {e}")
            return

        if not state.in_squat:
            state.squat_start = t
            state.in_squat = True
            log("⏳ POSSIVEL_DEFECACAO iniciada")
            mqtt_publish(client, TOPIC_STATE, "POSSIVEL_DEFECACAO")
        else:
            if (t - state.squat_start) >= SQUAT_MIN_S:
                log("💩 DEFECANDO (mantida postura)")
                mqtt_publish(client, TOPIC_STATE, "DEFECANDO")

        state.roi_anchor = roi_from_bbox(box, img.shape, pad=20)
        # baseline rápida (opcional) – não é usada para confirmar resíduo
        roi_quick = safe_roi(img, state.roi_anchor)
        state.bg_roi = None if roi_quick is None else roi_quick.copy()

    else:
        if state.in_squat:
            state.in_squat = False
            log("🔍 Saída da postura → aguardando confirmação de resíduo")
            mqtt_publish(client, TOPIC_STATE, "AGUARDANDO_CONFIRMACAO")
            threading.Thread(target=confirm_residue_window, args=(), daemon=True).start()

# ---------- RESIDUE ----------
def roi_from_bbox(b, frame_shape, pad=20):
    x1, y1, x2, y2 = [int(v) for v in b]
    h, w = frame_shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1r = max(0, int(cx - 90) - pad)
    x2r = min(w, int(cx + 90) + pad)
    y1r = max(0, int(y2) - 20)
    y2r = min(h, int(y2 + 120) + pad)
    return (x1r, y1r, x2r - x1r, y2r - y1r)

def residue_blob(bg_roi, cur_roi):
    # Tolerante a entradas inválidas
    if bg_roi is None or cur_roi is None:
        return None
    if bg_roi.size == 0 or cur_roi.size == 0:
        return None
    try:
        g0 = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return None

    g0 = cv2.GaussianBlur(g0, (5,5), 0)
    g1 = cv2.GaussianBlur(g1, (5,5), 0)
    d  = cv2.absdiff(g1, g0)
    _, th = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < RES_MIN_A:
            continue
        x,y,w,h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ratio = min(w,h)/max(w,h)
        if ratio < 0.25:
            continue
        if area > best_area:
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
            best = (x,y,w,h,(cx,cy),area)
            best_area = area
    return best

def confirm_residue_window():
    start = time.time()
    last_ok = 0.0
    best_centroid = None
    best_area = 0

    if state.roi_anchor is None:
        log("⚠️ Sem roi_anchor ao entrar na janela de confirmação")
        mqtt_publish(client, TOPIC_STATE, "IDLE")
        return

    log("🔍 Iniciando janela de confirmação de resíduo")
    baseline = None  # baseline capturada NA PRÓPRIA janela

    while time.time() - start <= RES_WIN_S:
        try:
            img = fetch_snapshot()
        except Exception as e:
            log(f"⚠️ snapshot fail (confirm window): {e}")
            time.sleep(max(0.1, 1.0 / max(1.0, SNAP_FPS)))
            continue

        cur_roi = safe_roi(img, state.roi_anchor)
        if cur_roi is None or cur_roi.size == 0:
            time.sleep(max(0.1, 1.0 / max(1.0, SNAP_FPS)))
            continue

        if baseline is None:
            baseline = cur_roi.copy()
            state.residue_bg = baseline.copy()   # guarda para monitoramento posterior
            log("📸 Baseline capturada para janela de resíduo")
            time.sleep(max(0.1, 1.0 / max(1.0, SNAP_FPS)))
            continue

        blob = residue_blob(baseline, cur_roi)
        if blob:
            bx, by, bw, bh, (cx, cy), area = blob
            if last_ok == 0.0:
                last_ok = time.time()
                # centroid em coords globais:
                x, y, w, h = state.roi_anchor
                best_centroid = (x + cx, y + cy)
                best_area = area
                log(f"🟡 possível resíduo: area={area}")
            elif time.time() - last_ok >= RES_STATIC_S:
                # ✅ CONFIRMADO
                state.poop_present = True
                state.poop_roi = state.roi_anchor
                state.poop_centroid = best_centroid

                log("✅ Resíduo confirmado (poop detectado)")
                mqtt_publish(client, TOPIC_STATE, "DEFECACAO_CONFIRMADA")
                mqtt_publish(client, TOPIC_POOP, {
                    "value": True,
                    "zone": ZONE,
                    "centroid": [int(best_centroid[0]), int(best_centroid[1])],
                    "area": int(best_area),
                    "ts": now_iso(),
                })

                # Tentar vincular ao evento do Frigate
                evt_id = find_recent_poop_event_id(time.time())
                payload = {
                    "ts": now_iso(),
                    "camera": CAM,
                    "zone": ZONE,
                    "event_id": evt_id or "",
                }
                if evt_id:
                    base_api = f"{FR_PUBLIC}/api/events/{evt_id}"
                    payload["clip_url"]  = f"{base_api}/clip.mp4"
                    payload["thumb_url"] = f"{base_api}/thumbnail.jpg"
                    payload["snapshot"]  = f"{base_api}/snapshot.jpg"

                mqtt_publish(client, TOPIC_POOP_EVENT, payload)

                # (Opcional) criar evento próprio “poop_confirmed”
                event_id = frigate_create_event(CAM, label="poop_confirmed", sub_label="watcher:auto")
                if event_id:
                    clip_url, thumb_url = frigate_event_urls(event_id)
                    mqtt_publish(client, TOPIC_POOP_EVENT, {
                        "ts": now_iso(),
                        "camera": CAM,
                        "zone": ZONE,
                        "event_id": event_id,
                        "clip_url": clip_url,
                        "thumb_url": thumb_url,
                        "centroid": [int(best_centroid[0]), int(best_centroid[1])],
                        "area": int(best_area)
                    })

                # Começa a vigiar a limpeza
                threading.Thread(target=monitor_pooppresent, args=(), daemon=True).start()
                return
        else:
            if last_ok != 0.0:
                log("🔄 blob sumiu antes de RES_STATIC_S, resetando janela de estabilidade")
            last_ok = 0.0

        time.sleep(max(0.1, 1.0 / max(1.0, SNAP_FPS)))

    log("⏹️ Janela de confirmação terminou sem resíduo")
    mqtt_publish(client, TOPIC_STATE, "IDLE")

def monitor_pooppresent():
    miss = 0
    log("🧼 Monitorando limpeza do cocô...")
    while state.poop_present and state.poop_roi is not None:
        time.sleep(CHECK_INT_S)
        try:
            img = fetch_snapshot()
        except Exception:
            continue

        cur_roi = safe_roi(img, state.poop_roi)
        if cur_roi is None or cur_roi.size == 0:
            continue

        baseline = state.residue_bg if state.residue_bg is not None else state.bg_roi
        blob = residue_blob(baseline, cur_roi)
        if blob is None:
            miss += 1
        else:
            miss = 0

        if miss >= 3:
            state.poop_present = False
            log("🧹 Cocô removido → estado IDLE")
            mqtt_publish(client, TOPIC_POOP, {"value": False, "zone": ZONE, "ts": now_iso()})
            mqtt_publish(client, TOPIC_STATE, "IDLE")
            return

# ---------- MQTT bootstrap ----------
try:
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()

if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)

client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
client.loop_start()

log("[ellie-watcher] running with:", f"camera={CAM}", f"zone={ZONE}", f"mqtt={MQTT_HOST}:{MQTT_PORT}")
log(f"Heurística: squat_thr={SQUAT_THR} min_dur={SQUAT_MIN_S}s")

if ENABLE_DEBUG_WATCHER:
    log(f"All helper variables: SQUAT_THR={SQUAT_THR}, SQUAT_MIN_S={SQUAT_MIN_S}, RES_WIN_S={RES_WIN_S}, RES_MIN_A={RES_MIN_A}, RES_STATIC_S={RES_STATIC_S}, SNAP_FPS={SNAP_FPS}, CHECK_INT_S={CHECK_INT_S}, ENABLE_DEBUG_WATCHER={ENABLE_DEBUG_WATCHER}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass