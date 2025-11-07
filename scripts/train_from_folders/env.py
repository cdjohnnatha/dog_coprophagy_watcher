import os
import sys
import io
import threading
from contextlib import contextmanager


def set_env_defaults() -> None:
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "loglevel;quiet")
    os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
    


def _silence_ffmpeg_native_logs() -> None:
    """Force FFmpeg (libav*) to AV_LOG_QUIET via ctypes, if available."""
    try:
        import ctypes
        from ctypes.util import find_library
        from pathlib import Path
    except Exception:
        return

    AV_LOG_QUIET = -8
    candidates = []
    lib_path = find_library("avutil")
    if lib_path:
        candidates.append(lib_path)
    candidates.extend([
        "libavutil.so", "libavutil.so.58", "libavutil.so.57",
        "libavutil.dylib", "libavutil.58.dylib", "libavutil.57.dylib",
        "avutil-58.dll", "avutil-57.dll",
    ])

    # Try to find libraries shipped within OpenCV wheels
    try:
        import cv2 as _cv2_for_paths  # type: ignore

        cv2_dir = Path(_cv2_for_paths.__file__).resolve().parent
        for pattern in ("libavutil*.dylib", "libavutil*.so", "avutil-*.dll"):
            for path in cv2_dir.glob(pattern):
                candidates.append(str(path))
        for pattern in ("libavutil*.dylib", "libavutil*.so", "avutil-*.dll"):
            for path in cv2_dir.rglob(pattern):
                candidates.append(str(path))
    except Exception:
        pass

    for cand in candidates:
        try:
            avutil = ctypes.CDLL(cand)
        except OSError:
            continue
        try:
            avutil.av_log_set_level.argtypes = [ctypes.c_int]  # type: ignore[attr-defined]
            avutil.av_log_set_level.restype = None  # type: ignore[attr-defined]
            avutil.av_log_set_level(AV_LOG_QUIET)  # type: ignore[attr-defined]
            break
        except Exception:
            continue


class FFmpegLogSuppressor:
    _lock = threading.Lock()
    _counter = 0
    _capturing = False

    def __enter__(self):
        if FFmpegLogSuppressor._capturing:
            return self
        FFmpegLogSuppressor._capturing = True
        self._orig_stderr = sys.stderr
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        out = sys.stderr.getvalue()
        sys.stderr = self._orig_stderr
        FFmpegLogSuppressor._capturing = False
        if out:
            count = out.count("SEI type")
            if count:
                with FFmpegLogSuppressor._lock:
                    FFmpegLogSuppressor._counter += count

    @staticmethod
    def summary() -> None:
        c = FFmpegLogSuppressor._counter
        if c:
            print(f"⚠️  Avisos SEI suprimidos: {c} ocorrências (vídeos com metadados truncados)")


@contextmanager
def suppress_stderr():
    """Silence OS-level STDERR (fd=2) while the context is active."""
    try:
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        try:
            os.dup2(saved_stderr_fd, stderr_fd)
        finally:
            os.close(saved_stderr_fd)


@contextmanager
def preimport_silence():
    """
    Silence STDERR during imports of cv2/tensorflow to avoid FFmpeg noise.
    Use at the very beginning of the CLI before importing heavy libs.
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def configure_environment() -> None:
    """Set environment defaults and attempt native ffmpeg silence."""
    set_env_defaults()
    _silence_ffmpeg_native_logs()


