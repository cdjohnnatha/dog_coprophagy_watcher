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

# --- EPISÓDIO / DEDUP ---
EP_COOLDOWN_S   = float(os.getenv("EP_COOLDOWN_S", "180"))  # mesclar "cocô partido" num só
LEAVE_TIMEOUT_S = float(os.getenv("LEAVE_TIMEOUT_S", "8"))  # cão precisa "sumir" X s p/ fechar episódio
MERGE_RADIUS_PX = int(os.getenv("MERGE_RADIUS_PX", "90"))   # distância p/ considerar o mesmo evento

# --- ANTI-PEE / SHAPE ---
MIN_BLOB_W      = int(os.getenv("MIN_BLOB_W", "14"))
MIN_BLOB_H      = int(os.getenv("MIN_BLOB_H", "14"))
MIN_CIRCULARITY = float(os.getenv("MIN_CIRCULARITY", "0.28"))  # 4πA/P²
MIN_SOLIDITY    = float(os.getenv("MIN_SOLIDITY", "0.65"))     # A/convexHull
HSV_S_MIN       = int(os.getenv("HSV_S_MIN", "25"))            # saturação mínima (poça molhada costuma ser baixa)
SAT_MONO_GLOBAL = int(os.getenv("SAT_MONO_GLOBAL", "12"))   # S médio abaixo disso => modo P&B/IR
TEX_MIN_MONO    = float(os.getenv("TEX_MIN_MONO", "25"))    # variância de Laplaciano mínima (textura)
SPEC_MAX_MONO   = float(os.getenv("SPEC_MAX_MONO", "0.18")) # fração máx. de pixels muito claros (reflexo/poça)
# --- PARADO DE VERDADE ---
MOTIONLESS_MIN   = int(os.getenv("MOTIONLESS_MIN", "3"))
SPEED_MAX_STILL  = float(os.getenv("SPEED_MAX_STILL", "0.15"))

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

def is_monochrome_roi(roi_bgr, sat_thresh=SAT_MONO_GLOBAL):
    """Retorna True se a ROI aparenta estar em P&B/IR (saturação média muito baixa)."""
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s_mean = float(cv2.mean(hsv[:, :, 1])[0])
    return s_mean < sat_thresh

def frigate_create_event(camera: str, label: str, sub_label: str | None = None) -> str | None:
    """
    Em versões recentes do Frigate, a criação de eventos via API (POST /api/events)
    não é suportada — apenas leitura. Em vez disso, retornamos None.
    """
    log("[frigate] skipping create_event (API não permite POST /api/events)")
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

def find_nearby_dog_event_id(ts_confirm: float) -> str | None:
    """Procura um evento DOG com clipe perto do horário da confirmação."""
    try:
        after = int(ts_confirm - 45)   # janelinha um pouco maior pra garantir
        before = int(ts_confirm + 120)
        base = f"{FR_URL}/api/events?camera={CAM}&label=dog&has_clip=1&after={after}&before={before}&limit=15"
        # filtro por zona ajuda a pegar o momento certo
        if ZONE:
            base += f"&zone={ZONE}"
        r = requests.get(base, timeout=3.0)
        r.raise_for_status()
        items = r.json() if isinstance(r.json(), list) else []
        if not items:
            return None
        # prioriza eventos que terminaram perto da confirmação
        def score(ev):
            st = float(ev.get("start_time", 0))
            et = float(ev.get("end_time", st))
            ref = et if et > 0 else st
            return abs(ts_confirm - ref)
        items.sort(key=score)
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
    episode_active: bool = False
    episode_started_ts: float = 0.0
    last_confirm_ts: float = 0.0
    last_confirm_centroid: tuple | None = None
    mono_flag: bool = False

state = TrackState()
last_seen_dog_ts = 0.0  # global simples p/ saber quando "sumiu"

# ---------- MQTT ----------
def on_connect(c, userdata, flags, rc, properties=None):
    log("🔌 MQTT connected", rc)
    c.subscribe(TOPIC_EVENTS, qos=0)
    mqtt_publish(c, "home/ellie/health", {"ok": True, "ts": now_iso(), "camera": CAM, "zone": ZONE})

def on_message(client, _userdata, msg):
    global last_seen_dog_ts
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

    # score (parado de verdade)
    ratio = float(after.get("ratio", 1.0))
    stationary = bool(after.get("stationary", False))
    speed = float(after.get("current_estimated_speed", 0.0))
    motionless = int(after.get("motionless_count", 0))

    ratio_term = max(0.0, min(1.0, (0.70 - ratio) / 0.20))
    really_still = stationary and (motionless >= MOTIONLESS_MIN) and (speed <= SPEED_MAX_STILL)
    stationary_term = 1.0 if really_still else 0.0

    speed_term_raw = max(0.0, min(1.0, 1.0 - min(speed, 1.0)))
    speed_term = 0.0 if speed > SPEED_MAX_STILL else speed_term_raw

    score = 0.55 * ratio_term + 0.30 * stationary_term + 0.15 * speed_term

    log(f"🐶 Zones={zones}, ratio={ratio:.2f}, stationary={stationary}, motionless={motionless}, "
        f"speed={speed:.2f}, still={really_still}, score={score:.3f}")

    state.last_ts = time.time()
    last_seen_dog_ts = state.last_ts
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
            # abrir episódio se ainda não houver
            if not state.episode_active:
                state.episode_active = True
                state.episode_started_ts = t
        else:
            if (t - state.squat_start) >= SQUAT_MIN_S:
                log("💩 DEFECANDO (mantida postura)")
                mqtt_publish(client, TOPIC_STATE, "DEFECANDO")

        state.roi_anchor = roi_from_bbox(box, img.shape, pad=20)
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

def residue_blob(bg_roi, cur_roi, monochrome=False):
    """Compara bg_roi vs cur_roi e retorna melhor contorno compatível com 'cocô' (anti-xixi)."""
    if bg_roi is None or cur_roi is None or bg_roi.size == 0 or cur_roi.size == 0:
        return None
    try:
        # força mesmo tamanho
        if bg_roi.shape[:2] != cur_roi.shape[:2]:
            cur_roi = cv2.resize(cur_roi, (bg_roi.shape[1], bg_roi.shape[0]), interpolation=cv2.INTER_AREA)

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
        if w < MIN_BLOB_W or h < MIN_BLOB_H:
            continue

        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        circ = 4.0*np.pi*area/(per*per)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if hull is not None else 0.0
        solidity = (area / hull_area) if hull_area > 0 else 0.0

                # métricas geométricas básicas
        if circ < MIN_CIRCULARITY:
            continue
        if solidity < MIN_SOLIDITY:
            continue

        mask = np.zeros(cur_roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        if not monochrome:
            # em modo colorido, usa saturação para filtrar poça de xixi
            hsv = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2HSV)
            mean_s = cv2.mean(hsv[:, :, 1], mask=mask)[0]
            if mean_s < HSV_S_MIN:
                continue
        else:
            # em P&B/IR: use textura + highlights para diferenciar
            gray = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)

            # textura: cocô costuma ter bordas/ruído de alta frequência
            var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            if var_lap < TEX_MIN_MONO:
                # muito "liso" => tende a ser poça/reflexo
                continue

            # highlights especulares: poça molhada brilha no IR
            area_mask = max(1, int(cv2.countNonZero(mask)))
            bright = cv2.countNonZero(cv2.bitwise_and((gray > 235).astype(np.uint8)*255, mask))
            frac_bright = bright / float(area_mask)
            if frac_bright > SPEC_MAX_MONO:
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
             # detecta se a ROI está com saturação muito baixa (modo IR/P&B)
            try:
                state.mono_flag = is_monochrome_roi(baseline)
            except Exception:
                state.mono_flag = False
            continue

        blob = residue_blob(baseline, cur_roi, monochrome=state.mono_flag)
        if blob:
            bx, by, bw, bh, (cx, cy), area = blob
            if last_ok == 0.0:
                last_ok = time.time()
                x, y, w, h = state.roi_anchor
                best_centroid = (x + cx, y + cy)  # coords globais
                best_area = area
                log(f"🟡 possível resíduo: area={area}")
            elif time.time() - last_ok >= RES_STATIC_S:
                centroid_abs = best_centroid

                # --- DEDUP por episódio: mescla confirmações próximas no tempo/espaço
                if state.episode_active and state.last_confirm_ts > 0:
                    dt = time.time() - state.last_confirm_ts
                    dist = 1e9
                    if state.last_confirm_centroid is not None:
                        dx = centroid_abs[0] - state.last_confirm_centroid[0]
                        dy = centroid_abs[1] - state.last_confirm_centroid[1]
                        dist = (dx*dx + dy*dy) ** 0.5
                    if dt <= EP_COOLDOWN_S and dist <= MERGE_RADIUS_PX:
                        log(f"➿ Mesclado ao mesmo episódio (dt={dt:.1f}s, dist={int(dist)}px)")
                        state.poop_present = True
                        state.poop_roi = state.roi_anchor
                        state.poop_centroid = centroid_abs
                        threading.Thread(target=monitor_pooppresent, args=(), daemon=True).start()
                        return

                # ✅ CONFIRMADO (novo)
                state.poop_present = True
                state.poop_roi = state.roi_anchor
                state.poop_centroid = centroid_abs
                state.last_confirm_ts = time.time()
                state.last_confirm_centroid = centroid_abs

                log("✅ Resíduo confirmado (poop detectado)")
                mqtt_publish(client, TOPIC_STATE, "DEFECACAO_CONFIRMADA")
                mqtt_publish(client, TOPIC_POOP, {
                    "value": True,
                    "zone": ZONE,
                    "centroid": [int(centroid_abs[0]), int(centroid_abs[1])],
                    "area": int(best_area),
                    "ts": now_iso(),
                })

                evt_id = find_nearby_dog_event_id(time.time())
                if not evt_id:
                    ts = int(time.time())
                    payload["export_url"] = f"{FR_URL}/api/export/{CAM}?start={ts-10}&end={ts+10}"

                payload = {
                    "ts": now_iso(),
                    "camera": CAM,
                    "zone": ZONE,
                    "event_id": evt_id or "",
                }

                # URLs diretas do Frigate (úteis p/ debug ou widgets custom)
                if evt_id:
                    base_api = f"{FR_URL}/api/events/{evt_id}"
                    payload["clip_url"]  = f"{base_api}/clip.mp4"
                    payload["thumb_url"] = f"{base_api}/thumbnail.jpg"
                    payload["snapshot"]  = f"{base_api}/snapshot.jpg"
                    payload["ha_clip"]   = f"/api/frigate/notifications/{evt_id}/clip.mp4"
                    payload["ha_thumb"]  = f"/api/frigate/notifications/{evt_id}/thumbnail.jpg"
                else:
                    # se não houver evento, usa placeholder vazio (não gera URL quebrada)
                    payload["clip_url"] = payload["thumb_url"] = payload["snapshot"] = ""
                    payload["ha_clip"] = payload["ha_thumb"] = ""

                mqtt_publish(client, TOPIC_POOP_EVENT, payload)

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

        baseline = state.residue_bg  # sempre a baseline da janela
        blob = residue_blob(baseline, cur_roi, monochrome=state.mono_flag)
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
    log(f"All helper variables: SQUAT_THR={SQUAT_THR}, SQUAT_MIN_S={SQUAT_MIN_S}, RES_WIN_S={RES_WIN_S}, "
        f"RES_MIN_A={RES_MIN_A}, RES_STATIC_S={RES_STATIC_S}, SNAP_FPS={SNAP_FPS}, CHECK_INT_S={CHECK_INT_S}, "
        f"ENABLE_DEBUG_WATCHER={ENABLE_DEBUG_WATCHER}, EP_COOLDOWN_S={EP_COOLDOWN_S}, "
        f"LEAVE_TIMEOUT_S={LEAVE_TIMEOUT_S}, MERGE_RADIUS_PX={MERGE_RADIUS_PX}, "
        f"MIN_CIRCULARITY={MIN_CIRCULARITY}, MIN_SOLIDITY={MIN_SOLIDITY}, HSV_S_MIN={HSV_S_MIN}, "
        f"MOTIONLESS_MIN={MOTIONLESS_MIN}, SPEED_MAX_STILL={SPEED_MAX_STILL}")

# ---------- LOOP principal: fecha episódio quando o cão some ----------
try:
    while True:
        time.sleep(1)
        if state.episode_active and (time.time() - last_seen_dog_ts) >= LEAVE_TIMEOUT_S:
            state.episode_active = False
except KeyboardInterrupt:
    pass