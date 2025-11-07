from typing import List, Tuple

import numpy as np
import tensorflow as tf
import os
from .video_io import load_video_clip


Item = Tuple[str, float, float]


def make_dataset(
    items: List[Item],
    num_frames: int,
    size: int,
    batch: int,
    shuffle: bool = True,
    parallel_reads: int | None = None,
    debug_io: bool = False,
    augment: bool = False,
):
    paths = np.array([p for p, _, _ in items], dtype=object)
    y_po = np.array([(-1 if np.isnan(v) else int(v)) for _, v, _ in items], dtype=np.int32)
    y_co = np.array([(-1 if np.isnan(v) else int(v)) for _, _, v in items], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y_po, y_co))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(256, len(items)))

    def _load(path, y1, y2):
        path = path.numpy().decode("utf-8")
        x = load_video_clip(path, num_frames=num_frames, target=size, debug=debug_io)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y1 = tf.reshape(y1, (1,))
        y2 = tf.reshape(y2, (1,))
        return x, y1, y2

    num_calls = parallel_reads or tf.data.AUTOTUNE
    ds = ds.map(
        lambda p, a, b: tf.py_function(
            func=_load, inp=[p, a, b], Tout=(tf.float32, tf.int32, tf.int32)
        ),
        num_parallel_calls=num_calls,
        deterministic=False,
    )

    def _ensure(x, y1, y2):
        x = tf.ensure_shape(x, (num_frames, size, size, 3))
        y = {"poop": tf.ensure_shape(y1, (1,)), "copro": tf.ensure_shape(y2, (1,))}
        return x, y

    ds = ds.map(_ensure, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Apply clip-consistent augmentations: same params across all frames
        def _augment(x, y):
            flip = tf.random.uniform(()) > 0.5
            x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
            # brightness delta in [-0.08, 0.08]
            bdelta = (tf.random.uniform(()) * 0.16) - 0.08
            x = tf.clip_by_value(x + bdelta, 0.0, 1.0)
            # contrast factor in [0.9, 1.1]
            c = 0.9 + tf.random.uniform(()) * 0.2
            # tf.image.adjust_contrast expects [H,W,C] or batched; map per frame
            x = tf.map_fn(lambda f: tf.image.adjust_contrast(f, c), x)
            x = tf.clip_by_value(x, 0.0, 1.0)
            return x, y

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    

    cache_root = os.path.join(os.getenv("ELLIE_OUTDIR", "."))
    cache_name = ("cache_train" if shuffle else "cache_val") + f"_f{num_frames}_s{size}"
    cache_dir = os.path.join(cache_root, cache_name)
    os.makedirs(cache_dir, exist_ok=True)
    ds = ds.cache(cache_dir)

    ds = ds.batch(batch)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


