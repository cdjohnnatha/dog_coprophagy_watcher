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

TOPIC_EVENTS = f"{MQTT_PREFIX}/events"
TOPIC_STATE  = "home/ellie/state"
TOPIC_POOP   = "home/ellie/poop_present"

# ---------- HELPERS ----------

def now_iso():
    """
    Returns the current timestamp in ISO 8601 format (UTC).

    Returns:
        str: Current timestamp as YYYY-MM-DDTHH:MM:SSZ
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def mqtt_publish(client, topic, payload):
    """
    Publishes a message to an MQTT topic. Automatically serializes dict/list payloads to JSON.

    Args:
        client: MQTT client instance
        topic (str): MQTT topic to publish to
        payload: Message payload (str, dict, or list)
    """
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, separators=(",", ":"))
    client.publish(topic, payload, qos=0, retain=False)

def fetch_snapshot():
    """
    Fetches the latest snapshot image from Frigate for the configured camera.

    Returns:
        np.ndarray: OpenCV image array (BGR format)

    Raises:
        requests.HTTPError: If the Frigate API request fails
    """
    # Stable Frigate endpoint for latest frame
    url = f"{FR_URL}/api/{CAM}/latest.jpg"
    r = requests.get(url, timeout=2.0)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

@dataclass
class TrackState:
    """
    Tracks the current state of dog detection and poop monitoring.

    Attributes:
        last_bbox (tuple | None): Last detected bounding box (x1,y1,x2,y2) in frame coordinates
        last_ts (float): Timestamp of last detection
        squat_start (float): Timestamp when squatting started
        in_squat (bool): Whether the dog is currently detected as squatting
        bg_roi (np.ndarray | None): Background ROI image for residue comparison
        roi_anchor (tuple | None): ROI coordinates (x,y,w,h) for residue detection
        poop_present (bool): Whether poop is currently detected as present
        poop_roi (tuple | None): ROI where poop presence is monitored
        poop_centroid (tuple | None): Centroid coordinates of detected poop
    """
    last_bbox: tuple | None = None   # (x1,y1,x2,y2) frame coordinates
    last_ts: float = 0.0
    squat_start: float = 0.0
    in_squat: bool = False
    bg_roi: np.ndarray | None = None
    roi_anchor: tuple | None = None  # (x,y,w,h) ROI for residue
    poop_present: bool = False
    poop_roi: tuple | None = None    # ROI where we check presence
    poop_centroid: tuple | None = None

state = TrackState()

# ---------- SQUAT DETECTION HEURISTICS ----------

def bbox_aspect(b):
    """
    Calculates the aspect ratio (height/width) of a bounding box.
    Lower ratios indicate squatting posture (dog appears shorter and wider).

    Args:
        b (tuple): Bounding box coordinates (x1,y1,x2,y2)

    Returns:
        float: Height to width ratio (lower values suggest squatting)
    """
    x1,y1,x2,y2 = b
    w = max(1, x2-x1)
    h = max(1, y2-y1)
    return h / w   # squat tends to lower (more squat -> smaller h/w ratio)

def bbox_center(b):
    """
    Calculates the center point of a bounding box.

    Args:
        b (tuple): Bounding box coordinates (x1,y1,x2,y2)

    Returns:
        tuple: Center coordinates (x, y)
    """
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def lower_body_mass_score(img, b):
    """
    Analyzes the "mass" (edge density) in the lower half of the bounding box.
    Higher scores indicate more visual content in the lower body region,
    which is characteristic of squatting posture.

    Args:
        img (np.ndarray): Input image
        b (tuple): Bounding box coordinates (x1,y1,x2,y2)

    Returns:
        float: Edge density score (0.0 to 1.0)
    """
    # ratio of "mass" (edges) in the lower half of the bbox
    x1,y1,x2,y2 = [int(v) for v in b]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return 0.0
    h = crop.shape[0]
    lower = crop[int(h*0.5):, :]
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 30, 80)
    score = edges.mean()/255.0  # 0..1
    return float(score)

def motion_score(prev_center, curr_center):
    """
    Calculates a motion score based on how stationary the dog appears.
    Stationary behavior (low movement) is indicative of squatting.

    Args:
        prev_center (tuple | None): Previous center position (x, y)
        curr_center (tuple | None): Current center position (x, y)

    Returns:
        float: Stationary score (higher = more stationary)
    """
    if prev_center is None or curr_center is None: return 0.0
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    dist = (dx*dx + dy*dy)**0.5
    # stationary ~ high score (inverse of velocity)
    return float(np.exp(-dist/10.0))

def squat_score(img, b, prev_b):
    """
    Combines multiple heuristics to determine if the dog is squatting.
    Weights: aspect ratio (45%), motion (35%), lower body mass (20%).

    Args:
        img (np.ndarray): Current frame image
        b (tuple): Current bounding box (x1,y1,x2,y2)
        prev_b (tuple | None): Previous bounding box

    Returns:
        float: Squat confidence score (0.0 to 1.0)
    """
    a = bbox_aspect(b)               # smaller is better (squat)
    m = motion_score(bbox_center(prev_b) if prev_b else None, bbox_center(b))
    l = lower_body_mass_score(img, b)
    # simple normalization:
    a_norm = max(0.0, min(1.0, (0.9 - a)))  # if h/w ~0.5-0.8 is interesting
    score = 0.45*a_norm + 0.35*m + 0.20*l
    return score

# ---------- RESIDUE DETECTION: "new object on the ground" ----------

def roi_from_bbox(b, frame_shape, pad=20):
    """
    Creates a region of interest (ROI) below the dog's bounding box where poop would likely appear.

    Args:
        b (tuple): Dog's bounding box (x1,y1,x2,y2)
        frame_shape (tuple): Frame dimensions (height, width)
        pad (int): Padding around the ROI

    Returns:
        tuple: ROI coordinates (x, y, width, height)
    """
    x1,y1,x2,y2 = [int(v) for v in b]
    h, w = frame_shape[:2]
    cx,cy = bbox_center(b)
    # ROI focused below the center (where the residue would fall)
    x1r = max(0, int(cx-90)-pad); x2r = min(w, int(cx+90)+pad)
    y1r = max(0, int(y2)-20);     y2r = min(h, int(y2+120)+pad)
    return (x1r,y1r,x2r-x1r,y2r-y1r)

def residue_blob(bg_roi, cur_roi):
    """
    Detects new blobs (potential poop) by comparing background and current ROI images.
    Uses image differencing and morphological operations to find static objects.

    Args:
        bg_roi (np.ndarray): Background ROI image (before defecation)
        cur_roi (np.ndarray): Current ROI image

    Returns:
        tuple | None: Best blob as (x,y,w,h,(cx,cy),area) or None if no suitable blob found
    """
    # diff + morphology → static blob
    g0 = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)
    g0 = cv2.GaussianBlur(g0, (5,5), 0)
    g1 = cv2.GaussianBlur(g1, (5,5), 0)
    d  = cv2.absdiff(g1, g0)
    _,th = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < RES_MIN_A: continue
        x,y,w,h = cv2.boundingRect(c)
        # reject very long things (toys/bottles tend to be long)
        if w==0 or h==0: continue
        ratio = min(w,h)/max(w,h)
        if ratio < 0.25:  # too thin
            continue
        if area > best_area:
            best_area = area
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
            best = (x,y,w,h,(cx,cy), area)
    return best  # None or tuple

# ---------- MQTT HANDLERS ----------

def on_connect(c, userdata, flags, rc, properties=None):
    """
    MQTT connection callback. Subscribes to Frigate events topic.
    """
    print("MQTT connected", rc)
    c.subscribe(TOPIC_EVENTS, qos=0)

def on_message(c, userdata, msg):
    """
    Processes incoming MQTT messages from Frigate.
    Handles dog detection events in the monitored zone and performs squat/residue analysis.

    Expected message format: Frigate event JSON with 'after' field containing detection data.
    """
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return
    # Frigate events: types new/update/end
    # We care when dog enters our ZONE and while tracked
    if data.get("after", {}).get("camera") != CAM:
        return
    label = data.get("after", {}).get("label")
    if label != "dog":
        return

    zones = data.get("after", {}).get("entered_zones", []) or []
    if ZONE not in zones:
        return

    # bboxes are in data.after.box (x,y,w,h) or data.after.top/left/bottom/right (varies by version)
    box = None
    after = data.get("after", {})
    if "box" in after and isinstance(after["box"], list) and len(after["box"])==4:
        x,y,w,h = after["box"]
        box = (int(x), int(y), int(x+w), int(y+h))
    elif all(k in after for k in ("top","left","bottom","right")):
        box = (int(after["left"]), int(after["top"]), int(after["right"]), int(after["bottom"]))
    if not box:
        return

    # Save bbox + timestamp for squat logic
    state.last_ts = time.time()
    prev_b = state.last_bbox
    state.last_bbox = box

    try:
        img = fetch_snapshot()
    except Exception:
        return

    score = squat_score(img, box, prev_b)
    # state "in_squat"
    t = time.time()
    if score >= SQUAT_THR:
        if not state.in_squat:
            state.squat_start = t
            state.in_squat = True
            mqtt_publish(client, TOPIC_STATE, "POSSIVEL_DEFECACAO")
        else:
            # already in squat; if passed minimum, mark DEFECANDO
            if (t - state.squat_start) >= SQUAT_MIN_S:
                mqtt_publish(client, TOPIC_STATE, "DEFECANDO")
        # background snapshot to compare later
        state.roi_anchor = roi_from_bbox(box, img.shape, pad=20)
        x,y,w,h = state.roi_anchor
        state.bg_roi = img[y:y+h, x:x+w].copy()

    else:
        # if was in squat and exited -> open window to confirm residue
        if state.in_squat:
            state.in_squat = False
            mqtt_publish(client, TOPIC_STATE, "AGUARDANDO_CONFIRMACAO")
            threading.Thread(target=confirm_residue_window, args=()).start()

def confirm_residue_window():
    """
    Monitors for poop residue after the dog stops squatting.
    Runs for a limited time window, checking for persistent blobs that indicate successful defecation.
    If confirmed, starts monitoring for when the poop is cleaned up.
    """
    start = time.time()
    last_ok = 0.0
    best_centroid = None
    best_area = 0
    if state.bg_roi is None or state.roi_anchor is None:
        return
    while time.time() - start <= RES_WIN_S:
        try:
            img = fetch_snapshot()
        except Exception:
            time.sleep(0.3); continue
        x,y,w,h = state.roi_anchor
        cur_roi = img[y:y+h, x:x+w]
        if cur_roi.size == 0:
            time.sleep(0.3); continue
        blob = residue_blob(state.bg_roi, cur_roi)
        if blob:
            bx,by,bw,bh,(cx,cy), area = blob
            # check "static" for a few seconds
            if last_ok==0.0:
                last_ok = time.time()
                best_centroid = (x+cx, y+cy)
                best_area = area
            elif time.time()-last_ok >= RES_STATIC_S:
                # confirmed!
                state.poop_present = True
                state.poop_roi = (x,y,w,h)
                state.poop_centroid = best_centroid
                mqtt_publish(client, TOPIC_STATE, "DEFECACAO_CONFIRMADA")
                mqtt_publish(client, TOPIC_POOP, {
                    "value": True,
                    "zone": ZONE,
                    "centroid": [int(best_centroid[0]), int(best_centroid[1])],
                    "area": int(best_area),
                    "ts": now_iso(),
                })
                # start monitor to turn off when it disappears
                threading.Thread(target=monitor_pooppresent, args=()).start()
                return
        else:
            last_ok = 0.0
        time.sleep(0.3)
    # window closed without confirming
    mqtt_publish(client, TOPIC_STATE, "IDLE")

def monitor_pooppresent():
    """
    Continuously monitors for poop cleanup. Periodically checks if the detected poop blob
    has disappeared, indicating it has been cleaned up. Uses a miss counter to avoid
    false negatives from temporary occlusions.
    """
    # re-check periodically if the blob disappeared (cleaning)
    miss = 0
    while state.poop_present and state.poop_roi is not None:
        time.sleep(CHECK_INT_S)
        try:
            img = fetch_snapshot()
        except Exception:
            continue
        x,y,w,h = state.poop_roi
        cur_roi = img[y:y+h, x:x+w]
        if cur_roi.size == 0:
            continue
        blob = residue_blob(state.bg_roi, cur_roi)
        if blob is None:
            miss += 1
        else:
            miss = 0
        if miss >= 3:
            state.poop_present = False
            mqtt_publish(client, TOPIC_POOP, {
                "value": False,
                "zone": ZONE,
                "ts": now_iso(),
            })
            mqtt_publish(client, TOPIC_STATE, "IDLE")
            return

# ---------- MQTT bootstrap (compat paho v1/v2) ----------
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

print("[ellie-watcher] running with:")
print(f"  camera={CAM} zone={ZONE} mqtt={MQTT_HOST}:{MQTT_PORT}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass