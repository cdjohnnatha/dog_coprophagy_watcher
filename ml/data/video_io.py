import time
from contextlib import nullcontext
from typing import Iterator

import cv2
import numpy as np

from .env import suppress_stderr


DEF_FRAMES = 16
DEF_SIZE = 224


def iter_video_frames_safe(video_path: str, max_secs: int = 12, max_total_reads: int = 600) -> Iterator[np.ndarray]:
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    try:
        reads = 0
        while True:
            if (time.time() - t0) > max_secs:
                break
            if reads >= max_total_reads:
                break
            ok, frame = cap.read()
            reads += 1
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def load_video_clip(path: str, num_frames: int = DEF_FRAMES, target: int = DEF_SIZE, debug: bool = False) -> np.ndarray:
    ctx = suppress_stderr() if not debug else nullcontext()
    with ctx:
        cap = cv2.VideoCapture(path)
        if not cap or not cap.isOpened():
            arr = np.zeros((num_frames, target, target, 3), dtype=np.float32)
            return arr

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idxs = np.linspace(0, max(0, total - 1), num_frames).astype(int)

        frames = []
        last = None
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = last if last is not None else np.zeros((target, target, 3), np.uint8)
            else:
                last = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (target, target), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()

    arr = np.stack(frames).astype(np.float32) / 255.0
    return arr


