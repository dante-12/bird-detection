#!/usr/bin/env python3

"""Drone Photo Mosaic / Bird View Pixel Processor (bvpp)

Author: Naoki

Given a directory of nadir (bird's-eye) images captured by a drone from multiple
positions and altitudes, this script uses camera/pose metadata (latitude,
longitude, altitude, yaw/heading, and field-of-view) to project each image onto
an approximate ground plane and mosaic them into a single image.

Assumptions / Notes:
- Ground is locally flat.
- Camera is approximately nadir-pointing (looking straight down). If pitch/roll
  are present in metadata, this script currently ignores them.
- Heading/yaw is interpreted as degrees clockwise from true north.
- Field-of-view: horizontal FOV in degrees. If only diagonal FOV is available,
  the script will approximate horizontal FOV based on image aspect ratio.
- Missing areas are filled with white.
- Output resolution is matched to the lowest-altitude image (highest spatial
  resolution), by taking the smallest meters-per-pixel among the inputs.

CLI:
  --in   directory; all files inside are processed
  --out  output image path

Dependencies:
  pip install pillow numpy exifread PyQt6
  sudo apt install exiftool libxcb-cursor0
"""

from __future__ import annotations

import argparse
import base64
import gc
import glob
import mimetypes
import io
import json
import math
import os
import posixpath
import subprocess
import sys
import threading
import webbrowser
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import time

try:
    import exifread  # type: ignore
except Exception:  # pragma: no cover
    exifread = None

try:
    import resource
except Exception:  # pragma: no cover
    resource = None

# Optional GUI dependencies (PyQt6). Import lazily unless --gui was requested;
# some incompatible Qt builds abort the interpreter at import time.
class _QtWidgetsStub:
    class QGraphicsView:
        pass

    class QMainWindow:
        pass


QT_AVAILABLE = False
if "--gui" in sys.argv:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        QT_AVAILABLE = True
    except ImportError:
        QtCore = QtGui = None
        QtWidgets = _QtWidgetsStub
else:
    QtCore = QtGui = None
    QtWidgets = _QtWidgetsStub

# Pillowで巨大画像を扱う（必要なら制限解除）
Image.MAX_IMAGE_PIXELS = None

SUPPORTED_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
MAX_JPG_TIF_DIM_PX = 64000
MEMORY_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024
ABS_REL_ALT_CHANGE_THRESHOLD_M = 0.15
AREA_MASK_MAX_DIM_PX = 4096
AREA_ERROR_SAFETY_FACTOR = 1.5

# Metadata keys we expect to use (as seen by exifread)
INSPECT_KEYS: Dict[str, Tuple[str, ...]] = {
    "lat": ("GPS Latitude", "XMP DJI:GPSLatitude", "XMP exif:GPSLatitude", "GPS Position"),
    "lon": ("GPS Longitude", "XMP DJI:GPSLongitude", "XMP exif:GPSLongitude", "GPS Position"),
    "alt": ("Relative Altitude", "XMP DJI:RelativeAltitude", "XMP DJI:RelativeAltitudeMeters", "GPS GPSAltitude"),
    "absolute_alt": ("Absolute Altitude", "XMP DJI:AbsoluteAltitude", "GPS GPSAltitude"),
    "flight_yaw": ("Flight Yaw Degree", "XMP DJI:FlightYawDegree", "XMP DJI:FlightYaw"),
    "gimbal_yaw": ("Gimbal Yaw Degree", "XMP DJI:GimbalYawDegree", "XMP DJI:GimbalYaw"),
    "gimbal_pitch": ("Gimbal Pitch Degree", "XMP DJI:GimbalPitchDegree", "XMP DJI:GimbalPitch"),
    "flight_pitch": ("Flight Pitch Degree", "XMP DJI:FlightPitchDegree", "XMP DJI:FlightPitch"),
    "flight_roll": ("Flight Roll Degree", "XMP DJI:FlightRollDegree", "XMP DJI:FlightRoll"),
    "flight_x_speed": ("Flight X Speed", "XMP DJI:FlightXSpeed"),
    "flight_y_speed": ("Flight Y Speed", "XMP DJI:FlightYSpeed"),
    "flight_z_speed": ("Flight Z Speed", "XMP DJI:FlightZSpeed"),
    "fov": (
        "XMP Camera:FieldOfView",
        "XMP DJI:CameraFOV",
        "Image FieldOfView",
        "XMP Camera:DiagonalFOV",
        "XMP DJI:DiagonalFOV",
        "Image DiagonalFOV",
        "EXIF FocalLength",
        "EXIF FocalLengthIn35mmFilm",
    ),
    "gimbal_roll": ("Gimbal Roll Degree", "XMP DJI:GimbalRollDegree", "XMP DJI:GimbalRoll"),
}


@dataclass(frozen=True)
class PhotoMeta:
    path: str
    w: int
    h: int
    lat_deg: float
    lon_deg: float
    alt_m: float
    yaw_deg: float  # clockwise from true north
    hfov_deg: float
    absolute_alt_m: Optional[float] = None
    flight_yaw_deg: Optional[float] = None
    gimbal_yaw_deg: Optional[float] = None
    pitch_deg: float = -90.0  # degrees; -90 is nadir
    flight_pitch_deg: Optional[float] = None
    gimbal_pitch_deg: Optional[float] = None
    gimbal_roll_deg: Optional[float] = None
    flight_roll_deg: Optional[float] = None
    flight_x_speed_mps: Optional[float] = None
    flight_y_speed_mps: Optional[float] = None
    flight_z_speed_mps: Optional[float] = None
    product_name: Optional[str] = None
    unique_camera_model: Optional[str] = None
    captured_at: Optional[str] = None


@dataclass(frozen=True)
class RenderOptions:
    undistort: bool = False
    k1: float = 0.1400
    k2: float = -0.3979
    k3: float = 0.4837
    alt_correction_m: float = 0.0  # effective altitude = Relative Altitude + alt_correction_m
    yaw_offset_deg: float = 0.0  # additional yaw correction (degrees, clockwise from north)
    yaw_invert: bool = False  # if True, use -yaw (legacy behavior)
    yaw_both: bool = False  # if True, yaw = Flight Yaw Degree + Gimbal Yaw Degree
    yaw_gimbal_only: bool = False  # if True, ignore Flight Yaw Degree and use only Gimbal Yaw Degree
    yaw_flight_only: bool = False  # if True, ignore Gimbal Yaw Degree and use only Flight Yaw Degree
    opacity_pct: float = 100.0  # 0-100; applies to all warped layers before compositing
    roi_warp: bool = True
    roi_margin_px: int = 8
    jpg_quality: int = 95  # JPEG quality (1-95 typical)
    png_compress_level: int = 6  # PNG compress_level (0-9)
    preview_max_dim: int = 2048  # GUI preview max width/height in pixels
    use_pitch: bool = False  # if True, use pitch_deg in geometry (tilt shift / cos correction)
    crop_optimize: bool = False  # if True, choose front image per-pixel by nearest image-center distance (save only)
    hfov_deg_override: Optional[float] = None  # if set, override EXIF-derived horizontal FOV for all images


class MemoryPressureError(RuntimeError):
    """Raised when the estimated memory demand is too close to the system limit."""


class SaveCancelledError(RuntimeError):
    """Raised when an interactive save job is cancelled by the user."""


def _format_bytes_hr(num_bytes: float) -> str:
    n = float(max(num_bytes, 0.0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


@dataclass
class SaveProgressState:
    start_time: float = field(default_factory=time.perf_counter)
    bytes_written: int = 0
    done: bool = False
    error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_bytes(self, count: int) -> None:
        with self.lock:
            self.bytes_written += max(int(count), 0)

    def mark_done(self, error: Optional[BaseException] = None) -> None:
        with self.lock:
            self.done = True
            self.error = None if error is None else str(error)

    def snapshot(self) -> Tuple[float, int, bool, Optional[str]]:
        with self.lock:
            return self.start_time, self.bytes_written, self.done, self.error


class ProgressFileWrapper:
    """Proxy file object that tracks bytes written by Pillow."""

    def __init__(self, raw: BinaryIO, state: SaveProgressState):
        self._raw = raw
        self._state = state

    def write(self, data) -> int:
        written = self._raw.write(data)
        count = len(data) if written is None else int(written)
        self._state.add_bytes(count)
        return written

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def fileno(self) -> int:
        raise io.UnsupportedOperation("ProgressFileWrapper does not expose fileno")

    def __getattr__(self, name: str):
        return getattr(self._raw, name)


def _yaw_mode_label(options: RenderOptions) -> str:
    if bool(getattr(options, "yaw_gimbal_only", False)):
        return "gimbal_only"
    if bool(getattr(options, "yaw_both", False)):
        return "flight+gimbal"
    return "flight_only"


def _build_parameter_overlay_lines(options: RenderOptions, out_path: str) -> List[str]:
    ext = os.path.splitext(out_path)[1].lower()
    items: List[Tuple[str, str]] = [
        ("out_format", ext.lstrip(".") or "auto"),
        ("alt_correction_m", f"{float(getattr(options, 'alt_correction_m', 0.0)):.2f}"),
        ("yaw_mode", _yaw_mode_label(options)),
        ("yaw_offset_deg", f"{float(getattr(options, 'yaw_offset_deg', 0.0)):.2f}"),
        ("yaw_invert", "on" if bool(getattr(options, "yaw_invert", False)) else "off"),
        ("opacity_pct", f"{float(getattr(options, 'opacity_pct', 100.0)):.1f}"),
        ("roi_warp", "on" if bool(getattr(options, "roi_warp", True)) else "off"),
        ("roi_margin_px", str(int(getattr(options, "roi_margin_px", 8)))),
        ("crop_optimize", "on" if bool(getattr(options, "crop_optimize", False)) else "off"),
        ("use_pitch", "on" if bool(getattr(options, "use_pitch", False)) else "off"),
        ("undistort", "on" if bool(getattr(options, "undistort", False)) else "off"),
    ]
    if bool(getattr(options, "undistort", False)):
        items.extend(
            [
                ("k1", f"{float(getattr(options, 'k1', 0.1400)):.4f}"),
                ("k2", f"{float(getattr(options, 'k2', -0.3979)):.4f}"),
                ("k3", f"{float(getattr(options, 'k3', 0.4837)):.4f}"),
            ]
        )
    if ext in (".jpg", ".jpeg"):
        items.append(("jpg_quality", str(int(getattr(options, "jpg_quality", 95)))))
    elif ext == ".png":
        items.append(("png_compress_level", str(int(getattr(options, "png_compress_level", 6)))))

    label_width = max((len(label) for label, _ in items), default=0)
    lines = ["bvpp params"]
    lines.extend(f"{label.ljust(label_width)} : {value}" for label, value in items)
    return lines


def _load_overlay_font(font_size: int):
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure_overlay_text(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    font,
    spacing: int,
) -> Tuple[int, int]:
    text = "\n".join(lines)
    try:
        left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
        return max(0, right - left), max(0, bottom - top)
    except Exception:
        widths: List[int] = []
        heights: List[int] = []
        for line in lines:
            sample = line if line else " "
            try:
                left, top, right, bottom = draw.textbbox((0, 0), sample, font=font)
                widths.append(max(0, right - left))
                heights.append(max(0, bottom - top))
            except Exception:
                widths.append(0)
                heights.append(0)
        total_h = sum(heights)
        if heights:
            total_h += spacing * (len(heights) - 1)
        return max(widths, default=0), total_h


def _annotate_image_with_parameters(img: Image.Image, options: RenderOptions, out_path: str) -> None:
    if img.width <= 0 or img.height <= 0:
        return

    lines = _build_parameter_overlay_lines(options, out_path)
    draw = ImageDraw.Draw(img) if img.mode == "RGBA" else ImageDraw.Draw(img, "RGBA")
    font_size = 11
    font = _load_overlay_font(font_size)
    margin = 8
    padding = 4
    spacing = 2
    text_w, text_h = _measure_overlay_text(draw, lines, font, spacing)
    box_w = text_w + (padding * 2)
    box_h = text_h + (padding * 2)

    x0 = margin
    y0 = max(0, img.height - margin - box_h)
    x1 = min(img.width - 1, x0 + box_w)
    y1 = min(img.height - 1, y0 + box_h)
    if x1 <= x0 or y1 <= y0:
        return

    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 255))
    draw.multiline_text(
        (x0 + padding, y0 + padding),
        "\n".join(lines),
        fill=(255, 255, 255, 255),
        font=font,
        spacing=spacing,
    )


def _save_progress_metrics(state: SaveProgressState) -> Tuple[float, int, float]:
    start_time, bytes_written, _, _ = state.snapshot()
    elapsed = max(time.perf_counter() - start_time, 0.0)
    speed_mb_s = 0.0 if elapsed <= 1e-6 else (bytes_written / (1024.0 * 1024.0)) / elapsed
    return elapsed, bytes_written, speed_mb_s


def _format_save_progress_text(state: SaveProgressState, prefix: str = "Saving...") -> str:
    elapsed, bytes_written, speed_mb_s = _save_progress_metrics(state)
    return (
        f"{prefix} elapsed {elapsed:.1f}s, "
        f"size {_format_bytes_hr(bytes_written)}, speed {speed_mb_s:.1f} MB/s"
    )


def _save_image_job(img: Image.Image, out_path: str, options: RenderOptions, state: SaveProgressState) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        _annotate_image_with_parameters(img, options, out_path)
        with open(out_path, "wb") as raw_fp:
            progress_fp = ProgressFileWrapper(raw_fp, state)
            _save_image_with_options(img, progress_fp, options, out_path_hint=out_path)
            progress_fp.flush()
        state.mark_done()
    except BaseException as exc:
        state.mark_done(exc)
        raise


def _read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.readline().strip()
    except Exception:
        return None


def _cgroup_memory_limit_bytes() -> Optional[int]:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read_first_line(path)
        if not raw or raw == "max":
            continue
        try:
            val = int(raw)
        except Exception:
            continue
        if val > 0 and val < (1 << 60):
            return val
    return None


def _cgroup_memory_current_bytes() -> Optional[int]:
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        raw = _read_first_line(path)
        if not raw:
            continue
        try:
            val = int(raw)
        except Exception:
            continue
        if val >= 0:
            return val
    return None


def _sysctl_int(name: str) -> Optional[int]:
    try:
        proc = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return int(proc.stdout.strip())
    except Exception:
        return None


def _macos_page_size_bytes() -> Optional[int]:
    try:
        size = int(os.sysconf("SC_PAGE_SIZE"))
        if size > 0:
            return size
    except Exception:
        pass
    return _sysctl_int("hw.pagesize")


def _macos_vm_stat_pages() -> Dict[str, int]:
    pages: Dict[str, int] = {}
    try:
        proc = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode != 0:
            return pages
    except Exception:
        return pages

    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        digits = "".join(ch for ch in raw_value if ch.isdigit())
        if not digits:
            continue
        pages[key.strip()] = int(digits)
    return pages


def _macos_mem_available_bytes() -> Optional[int]:
    page_size = _macos_page_size_bytes()
    if page_size is None:
        return None
    pages = _macos_vm_stat_pages()
    if not pages:
        return None
    available_pages = (
        pages.get("Pages free", 0)
        + pages.get("Pages inactive", 0)
        + pages.get("Pages speculative", 0)
    )
    return int(available_pages) * int(page_size)


def _system_mem_available_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass
    if sys.platform == "darwin":
        return _macos_mem_available_bytes()
    return None


def _system_mem_total_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass
    if sys.platform == "darwin":
        return _sysctl_int("hw.memsize")
    return None


def _process_rss_bytes() -> Optional[int]:
    if resource is None:
        return None
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss_kb) * 1024
    except Exception:
        return None


def _memory_headroom_bytes() -> Optional[int]:
    candidates: List[int] = []
    system_available = _system_mem_available_bytes()
    if system_available is not None:
        candidates.append(system_available)

    cgroup_limit = _cgroup_memory_limit_bytes()
    cgroup_current = _cgroup_memory_current_bytes()
    if cgroup_limit is not None and cgroup_current is not None:
        candidates.append(max(cgroup_limit - cgroup_current, 0))

    if not candidates:
        return None
    return min(candidates)


def _estimated_canvas_memory_bytes(canvas_w: int, canvas_h: int, out_path: str, use_crop_opt: bool) -> int:
    pixels = int(canvas_w) * int(canvas_h)
    total = pixels * 4  # RGBA base canvas
    if use_crop_opt:
        total += pixels * 4  # float32 best_dist2
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        total += pixels * 3  # RGB flatten buffer for JPEG save
    total += pixels // 8  # small Pillow/allocator slack for giant images
    return total


def _estimate_photo_peak_bytes(
    meta: PhotoMeta,
    canvas_w: int,
    canvas_h: int,
    mpp_out: float,
    options: RenderOptions,
) -> int:
    base = int(meta.w) * int(meta.h) * 4  # decoded RGBA source
    if not bool(getattr(options, "roi_warp", True)):
        roi_pixels = int(canvas_w) * int(canvas_h)
    else:
        scale = _compute_mpp(meta, options) / max(float(mpp_out), 1e-9)
        roi_w = min(canvas_w, max(1, int(math.ceil(meta.w * scale)) + 2 * int(getattr(options, "roi_margin_px", 8))))
        roi_h = min(canvas_h, max(1, int(math.ceil(meta.h * scale)) + 2 * int(getattr(options, "roi_margin_px", 8))))
        roi_pixels = roi_w * roi_h
    roi_rgba = roi_pixels * 4
    composite_tmp = roi_pixels * 8
    return base + roi_rgba + composite_tmp


def _ensure_memory_headroom(required_bytes: int, context: str, margin_bytes: int = MEMORY_SAFETY_MARGIN_BYTES) -> None:
    headroom = _memory_headroom_bytes()
    if headroom is None:
        return
    if required_bytes + margin_bytes <= headroom:
        return

    rss = _process_rss_bytes()
    shortage = required_bytes + margin_bytes - headroom
    if required_bytes > headroom:
        msg = (
            f"Memory check failed before {context}: estimated additional memory "
            f"{_format_bytes_hr(required_bytes)} exceeds available headroom "
            f"{_format_bytes_hr(headroom)} by {_format_bytes_hr(shortage)}."
        )
    else:
        msg = (
            f"Memory check failed before {context}: estimated additional memory "
            f"{_format_bytes_hr(required_bytes)} leaves less than the required safety margin "
            f"with available headroom {_format_bytes_hr(headroom)}."
        )
    if margin_bytes > 0:
        msg += f" Safety margin: {_format_bytes_hr(margin_bytes)}."
    if rss is not None:
        msg += f" Current process RSS peak: {_format_bytes_hr(rss)}."
    msg += " Reduce output size, disable crop optimization, or free system memory and try again."
    raise MemoryPressureError(msg)


def _effective_alt_m(meta: PhotoMeta, options: RenderOptions) -> float:
    # camera-to-target shortest distance approximation (includes altitude correction)
    eff = float(meta.alt_m) + float(getattr(options, "alt_correction_m", 0.0))
    # If camera is tilted, effective vertical height to the plane is eff*cos(tilt)
    # Apply this correction only when use_pitch is set
    if bool(getattr(options, "use_pitch", False)):
        tilt_deg = _tilt_from_nadir_deg(getattr(meta, "pitch_deg", -90.0))
        eff = eff * max(math.cos(math.radians(tilt_deg)), 0.01)
    return max(eff, 0.01)


def _tilt_from_nadir_deg(pitch_deg: float) -> float:
    """Return tilt angle away from nadir, in degrees (0=nadir)."""
    # Nadir is around -90. Tilt magnitude is deviation from -90.
    return float(pitch_deg) - (-90.0)


def _dms_to_deg(values, ref: str) -> float:
    # exifread returns Ratio objects; convert robustly.
    def _to_float(x) -> float:
        try:
            return float(x.num) / float(x.den)
        except Exception:
            return float(x)

    d = _to_float(values[0])
    m = _to_float(values[1])
    s = _to_float(values[2])
    deg = d + (m / 60.0) + (s / 3600.0)
    if ref in ("S", "W"):
        deg = -deg
    return deg


def _try_parse_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        # exifread Tag: str(tag) is human-readable; try numeric first
        if hasattr(v, "values"):
            # sometimes it's a list/tuple
            if isinstance(v.values, (list, tuple)) and len(v.values) == 1:
                return float(v.values[0])
        if hasattr(v, "num") and hasattr(v, "den"):
            return float(v.num) / float(v.den)
        s = str(v).strip()
        # Common cases: "-90.0", "-90", "-90/1", "-90 deg", "-90.0,0,0"
        if "," in s and all(p.strip().replace(".", "", 1).replace("-", "", 1).isdigit() for p in s.split(",")[:1]):
            # take first
            s = s.split(",")[0].strip()
        # strip unit suffix
        for suf in ("deg", "degree", "degrees"):
            if s.lower().endswith(suf):
                s = s[: -len(suf)].strip()
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a.strip()) / float(b.strip())
        return float(s)
    except Exception:
        return None


def _find_tag_key(tags: Dict[str, object], candidates: Iterable[str]) -> Optional[str]:
    """Find the first matching tag key.

    exifread's tag naming can differ by file/vendor (e.g., XMP DJI:FlightYawDegree
    vs XMP:FlightYawDegree vs Image Flight Yaw Degree). This searches first for
    exact hits, then for case-insensitive substring hits.
    """
    # exact
    for c in candidates:
        if c in tags:
            return c

    # case-insensitive substring
    lower_map = {k.lower(): k for k in tags.keys()}
    for c in candidates:
        cl = c.lower()
        for tk_l, tk in lower_map.items():
            if cl in tk_l:
                return tk
    return None


def _extract_first_float(tags: Dict[str, object], keys: Iterable[str]) -> Optional[float]:
    # exact first
    for k in keys:
        if k in tags:
            v = _try_parse_float(tags.get(k))
            if v is not None:
                return float(v)

    # fuzzy fallback (namespace/key variations)
    k2 = _find_tag_key(tags, keys)
    if k2 is not None:
        v = _try_parse_float(tags.get(k2))
        if v is not None:
            return float(v)

    return None


def _extract_first_text(tags: Dict[str, object], keys: Iterable[str]) -> Optional[str]:
    # exact first
    for k in keys:
        if k in tags:
            s = str(tags.get(k)).strip()
            if s:
                return s

    # fuzzy fallback (namespace/key variations)
    k2 = _find_tag_key(tags, keys)
    if k2 is not None:
        s = str(tags.get(k2)).strip()
        if s:
            return s

    return None


def _short(v: object, n: int = 60) -> str:
    s = str(v)
    s = " ".join(s.split())
    if len(s) > n:
        return s[: n - 3] + "..."
    return s


def _read_exiftool_json(path: str) -> Dict[str, object]:
    """Read tags by invoking exiftool and returning a flat dict.

    This is used as a fallback because exifread does not always expose DJI flight/gimbal
    XMP fields with stable tag names.
    """
    # -n: numeric output, -j: JSON, -G: include group names, -s: short tag names
    # We still keep the original (non-grouped) tag labels too for compatibility.
    cmd = [
        "exiftool",
        "-n",
        "-j",
        "-G",
        "-s",
        path,
    ]
    try:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("exiftool not found; install exiftool or pip install exifread") from e

    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "exiftool failed")

    arr = json.loads(p.stdout or "[]")
    if not arr:
        return {}

    raw = arr[0]

    # raw contains keys like:
    #   "XMP-drone-dji:FlightYawDegree": -0.2
    #   "XMP-drone-dji:GimbalPitchDegree": -89.9
    #   "XMP:RelativeAltitude": 42.0
    #   "Composite:GPSPosition": "35. ... 139. ..."
    # plus many others.
    out: Dict[str, object] = {}

    # Keep everything as-is too (for debugging/fuzzy matching)
    for k, v in raw.items():
        if k == "SourceFile":
            continue
        out[k] = v

    # Add normalized aliases that our existing extractors/inspect expect.
    def _first(*keys: str) -> Optional[object]:
        for kk in keys:
            if kk in raw:
                return raw[kk]
        return None

    # Flight/Gimbal
    v = _first(
        "XMP-drone-dji:FlightYawDegree",
        "XMP-drone-dji:FlightYawDegree#",
        "XMP:FlightYawDegree",
        "FlightYawDegree",
    )
    if v is not None:
        out["Flight Yaw Degree"] = v

    v = _first(
        "XMP-drone-dji:GimbalYawDegree",
        "XMP-drone-dji:GimbalYawDegree#",
        "XMP:GimbalYawDegree",
        "GimbalYawDegree",
    )
    if v is not None:
        out["Gimbal Yaw Degree"] = v

    v = _first(
        "XMP-drone-dji:GimbalPitchDegree",
        "XMP-drone-dji:GimbalPitchDegree#",
        "XMP:GimbalPitchDegree",
        "GimbalPitchDegree",
    )
    if v is not None:
        out["Gimbal Pitch Degree"] = v

    v = _first(
        "XMP-drone-dji:FlightPitchDegree",
        "XMP-drone-dji:FlightPitchDegree#",
        "XMP:FlightPitchDegree",
        "FlightPitchDegree",
    )
    if v is not None:
        out["Flight Pitch Degree"] = v

    v = _first(
        "XMP-drone-dji:FlightRollDegree",
        "XMP-drone-dji:FlightRollDegree#",
        "XMP:FlightRollDegree",
        "FlightRollDegree",
    )
    if v is not None:
        out["Flight Roll Degree"] = v

    for axis in ("X", "Y", "Z"):
        v = _first(
            f"XMP-drone-dji:Flight{axis}Speed",
            f"XMP-drone-dji:Flight{axis}Speed#",
            f"XMP:Flight{axis}Speed",
            f"Flight{axis}Speed",
        )
        if v is not None:
            out[f"Flight {axis} Speed"] = v

    # Altitude
    v = _first(
        "XMP:AbsoluteAltitude",
        "XMP-drone-dji:AbsoluteAltitude",
        "AbsoluteAltitude",
    )
    if v is not None:
        out["Absolute Altitude"] = v

    v = _first(
        "XMP:RelativeAltitude",
        "XMP-drone-dji:RelativeAltitude",
        "RelativeAltitude",
    )
    if v is not None:
        out["Relative Altitude"] = v

    # GPS
    v = _first("Composite:GPSPosition", "GPSPosition")
    if v is not None:
        out["GPS Position"] = v

    v = _first("Composite:GPSLatitude", "GPSLatitude")
    if v is not None:
        out["GPS Latitude"] = v

    v = _first("Composite:GPSLongitude", "GPSLongitude")
    if v is not None:
        out["GPS Longitude"] = v

    # FOV tags if present
    v = _first("XMP:FieldOfView", "XMP-camera:FieldOfView", "FieldOfView")
    if v is not None:
        out["XMP Camera:FieldOfView"] = v

    v = _first("XMP:DiagonalFOV", "XMP-camera:DiagonalFOV", "DiagonalFOV")
    if v is not None:
        out["XMP Camera:DiagonalFOV"] = v

    v = _first("XMP-drone-dji:ProductName", "XMP:ProductName", "ProductName", "Product Name")
    if v is not None:
        out["Product Name"] = v

    v = _first(
        "XMP-exifEX:UniqueCameraModel",
        "XMP-exif:UniqueCameraModel",
        "EXIF:UniqueCameraModel",
        "UniqueCameraModel",
        "Unique Camera Model",
    )
    if v is not None:
        out["Unique Camera Model"] = v

    v = _first(
        "Composite:SubSecDateTimeOriginal",
        "EXIF:DateTimeOriginal",
        "EXIF:CreateDate",
        "EXIF:ModifyDate",
        "DateTimeOriginal",
        "CreateDate",
        "ModifyDate",
    )
    if v is not None:
        out["Date Time Original"] = v

    return out


def _read_exif_tags(path: str) -> Dict[str, object]:
    """Read tags.

    Prefer exifread when available, but merge exiftool output so DJI flight/gimbal
    fields are available reliably.
    """
    tags: Dict[str, object] = {}

    if exifread is not None:
        try:
            with open(path, "rb") as f:
                tags = exifread.process_file(f, details=False)
        except Exception:
            tags = {}

    # If key fields are missing, merge in exiftool-derived tags.
    need_merge = True
    if tags:
        # If any of these exist, still merge because exifread often misses DJI tags.
        need_merge = True

    if need_merge:
        try:
            et = _read_exiftool_json(path)
            # exifread keys and exiftool keys won't collide often; if they do, prefer exifread.
            for k, v in et.items():
                if k not in tags:
                    tags[k] = v
        except Exception:
            # exiftool not available or failed; keep exifread-only
            pass

    return tags


def _extract_lat_lon(tags: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    # Standard EXIF GPS
    lat, lon = None, None
    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")
    if lat_tag is not None and lat_ref is not None:
        try:
            lat = _dms_to_deg(lat_tag.values, str(lat_ref))
        except Exception:
            lat = None
    if lon_tag is not None and lon_ref is not None:
        try:
            lon = _dms_to_deg(lon_tag.values, str(lon_ref))
        except Exception:
            lon = None

    # DJI/XMP often exposes decimal directly
    if lat is None:
        lat = _extract_first_float(
            tags,
            (
                "GPS Latitude",
                "XMP DJI:GPSLatitude",
                "XMP exif:GPSLatitude",
            ),
        )
    if lon is None:
        lon = _extract_first_float(
            tags,
            (
                "GPS Longitude",
                "XMP DJI:GPSLongitude",
                "XMP exif:GPSLongitude",
            ),
        )

    # GPS Position like "35.123456 139.123456" or "35.123456,139.123456"
    if (lat is None or lon is None) and "GPS Position" in tags:
        s = str(tags.get("GPS Position")).strip()
        # normalize separators
        s2 = s.replace(";", " ").replace(",", " ")
        parts = [p for p in s2.split() if p]
        if len(parts) >= 2:
            try:
                if lat is None:
                    lat = float(parts[0])
                if lon is None:
                    lon = float(parts[1])
            except Exception:
                pass

    return lat, lon


def _extract_alt(tags: Dict[str, object]) -> Optional[float]:
    # Prefer DJI/XMP relative altitude when provided
    rel_alt = _extract_first_float(
        tags,
        (
            "Relative Altitude",
            "XMP DJI:RelativeAltitude",
            "XMP DJI:RelativeAltitudeMeters",
        ),
    )
    if rel_alt is not None:
        return float(rel_alt)

    # Fallback to EXIF GPS altitude
    alt_tag = tags.get("GPS GPSAltitude")
    alt_ref = tags.get("GPS GPSAltitudeRef")
    alt = _try_parse_float(alt_tag)
    if alt is None:
        return None
    # AltitudeRef 1 means below sea level
    if alt_ref is not None and str(alt_ref).strip() in ("1", "Below Sea Level"):
        alt = -alt
    return float(alt)


def _extract_absolute_alt(tags: Dict[str, object]) -> Optional[float]:
    absolute_alt = _extract_first_float(
        tags,
        (
            "Absolute Altitude",
            "XMP DJI:AbsoluteAltitude",
            "XMP:AbsoluteAltitude",
        ),
    )
    if absolute_alt is not None:
        return float(absolute_alt)

    alt_tag = tags.get("GPS GPSAltitude")
    alt_ref = tags.get("GPS GPSAltitudeRef")
    alt = _try_parse_float(alt_tag)
    if alt is None:
        return None
    if alt_ref is not None and str(alt_ref).strip() in ("1", "Below Sea Level"):
        alt = -alt
    return float(alt)


def _extract_yaw(tags: Dict[str, object], options: Optional[RenderOptions] = None) -> Optional[float]:
    """Extract yaw in degrees (clockwise from true north).

    Default behavior: yaw = flight_yaw.
    If options.yaw_both is True: yaw = flight_yaw + gimbal_yaw.
    If options.yaw_gimbal_only is True: yaw = gimbal_yaw.
    If options.yaw_flight_only is True: yaw = flight_yaw.
    Precedence: gimbal_only > yaw_both > flight_only/default.
    """
    gimbal = _extract_first_float(
        tags,
        (
            "Gimbal Yaw Degree",
            "XMP DJI:GimbalYawDegree",
            "XMP DJI:GimbalYaw",
        ),
    )

    flight = _extract_first_float(
        tags,
        (
            "Flight Yaw Degree",
            "XMP DJI:FlightYawDegree",
            "XMP DJI:FlightYaw",
        ),
    )

    if options is not None and bool(getattr(options, "yaw_gimbal_only", False)):
        if gimbal is not None:
            return float(gimbal) % 360.0
        return None

    if options is not None and bool(getattr(options, "yaw_both", False)):
        if flight is not None or gimbal is not None:
            yaw = float(flight or 0.0) + float(gimbal or 0.0)
            return yaw % 360.0
        return None

    # Default and --yaw-flight-only resolve to the same behavior.
    if flight is not None:
        return float(flight) % 360.0

    # If flight yaw is unavailable, fall back to gimbal yaw before generic fallback.
    if gimbal is not None:
        return float(gimbal) % 360.0

    if flight is not None or gimbal is not None:
        yaw = float(flight or 0.0) + float(gimbal or 0.0)
        return yaw % 360.0

    # Fallback: EXIF/GPX direction/bearing
    for k in (
        "GPS GPSImgDirection",
        "GPS Img Direction",
        "GPS GPSDestBearing",
        "GPS GPSTrack",
        "GPS GPSDirection",
    ):
        if k in tags:
            v = _try_parse_float(tags.get(k))
            if v is not None:
                return float(v) % 360.0

    return None


def _extract_flight_yaw(tags: Dict[str, object]) -> Optional[float]:
    return _extract_first_float(
        tags,
        (
            "Flight Yaw Degree",
            "XMP DJI:FlightYawDegree",
            "XMP DJI:FlightYaw",
        ),
    )


def _extract_gimbal_yaw(tags: Dict[str, object]) -> Optional[float]:
    return _extract_first_float(
        tags,
        (
            "Gimbal Yaw Degree",
            "XMP DJI:GimbalYawDegree",
            "XMP DJI:GimbalYaw",
        ),
    )


def _extract_gimbal_pitch(tags: Dict[str, object]) -> Optional[float]:
    return _extract_first_float(
        tags,
        (
            "Gimbal Pitch Degree",
            "XMP DJI:GimbalPitchDegree",
            "XMP DJI:GimbalPitch",
        ),
    )


def _extract_gimbal_roll(tags: Dict[str, object]) -> Optional[float]:
    return _extract_first_float(
        tags,
        (
            "Gimbal Roll Degree",
            "XMP DJI:GimbalRollDegree",
            "XMP DJI:GimbalRoll",
        ),
    )


def _extract_flight_pitch(tags: Dict[str, object]) -> Optional[float]:
    return _extract_first_float(
        tags,
        (
            "Flight Pitch Degree",
            "XMP DJI:FlightPitchDegree",
            "XMP DJI:FlightPitch",
        ),
    )


def _extract_flight_roll(tags: Dict[str, object]) -> Optional[float]:
    return _extract_first_float(
        tags,
        (
            "Flight Roll Degree",
            "XMP DJI:FlightRollDegree",
            "XMP DJI:FlightRoll",
        ),
    )


def _extract_flight_speed(tags: Dict[str, object], axis: str) -> Optional[float]:
    return _extract_first_float(
        tags,
        (
            f"Flight {axis} Speed",
            f"XMP DJI:Flight{axis}Speed",
            f"XMP:Flight{axis}Speed",
            f"Flight{axis}Speed",
        ),
    )


def _extract_pitch(tags: Dict[str, object]) -> Optional[float]:
    """Combine flight + gimbal pitch when present.

    DJI tags are often both present; summing is a pragmatic approximation.
    """
    fp = _extract_flight_pitch(tags)
    gp = _extract_gimbal_pitch(tags)
    if fp is None and gp is None:
        return None
    return float(fp or 0.0) + float(gp or 0.0)


def _extract_fov(tags: Dict[str, object], w: int, h: int) -> Optional[float]:
    # Prefer explicit FOV if available.
    for k in (
        "XMP Camera:FieldOfView",
        "XMP DJI:CameraFOV",
        "Image FieldOfView",
    ):
        if k in tags:
            v = _try_parse_float(tags.get(k))
            if v is not None and v > 0:
                return float(v)

    # Otherwise compute from focal length and sensor size if present
    # hfov = 2 * atan(sensor_width / (2 * focal_length))
    focal = None
    if "EXIF FocalLength" in tags:
        focal = _try_parse_float(tags.get("EXIF FocalLength"))

    # Sensor size is rarely in EXIF; if not present, cannot compute reliably.
    # As a fallback, attempt using 35mm equivalent focal length.
    focal35 = None
    if "EXIF FocalLengthIn35mmFilm" in tags:
        focal35 = _try_parse_float(tags.get("EXIF FocalLengthIn35mmFilm"))

    if focal35 is not None and focal35 > 0:
        # 35mm full-frame sensor width is 36mm.
        hfov = math.degrees(2.0 * math.atan(36.0 / (2.0 * focal35)))
        if hfov > 0:
            return hfov

    if focal is not None and focal > 0:
        # If we don't know sensor width, assume 1/2.3" ~ 6.17mm (common for older drones).
        # This is only a heuristic.
        sensor_width_mm = 6.17
        hfov = math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal)))
        if hfov > 0:
            return hfov

    return None


def _extract_capture_time(tags: Dict[str, object]) -> Optional[str]:
    value = _extract_first_text(
        tags,
        (
            "Date Time Original",
            "EXIF DateTimeOriginal",
            "Image DateTime",
            "EXIF:DateTimeOriginal",
            "EXIF:CreateDate",
            "EXIF:ModifyDate",
            "Composite:SubSecDateTimeOriginal",
            "DateTimeOriginal",
            "CreateDate",
            "ModifyDate",
        ),
    )
    if value is None:
        return None
    return str(value).strip()


def _approx_hfov_from_dfov(dfov_deg: float, w: int, h: int) -> float:
    # Convert diagonal FOV to horizontal FOV using aspect ratio.
    # tan(dfov/2)^2 = tan(hfov/2)^2 + tan(vfov/2)^2, and tan(vfov/2)=tan(hfov/2)/ar
    ar = w / float(h)
    t = math.tan(math.radians(dfov_deg) / 2.0)
    th = t / math.sqrt(1.0 + (1.0 / (ar * ar)))
    return math.degrees(2.0 * math.atan(th))


def _load_photo_meta(path: str, options: Optional[RenderOptions] = None) -> Optional[PhotoMeta]:
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        return None

    tags = _read_exif_tags(path)

    lat, lon = _extract_lat_lon(tags)
    alt = _extract_alt(tags)
    absolute_alt = _extract_absolute_alt(tags)
    yaw = _extract_yaw(tags, options)
    pitch = _extract_pitch(tags)
    flight_yaw = _extract_flight_yaw(tags)
    gimbal_yaw = _extract_gimbal_yaw(tags)
    flight_pitch = _extract_flight_pitch(tags)
    flight_roll = _extract_flight_roll(tags)
    flight_x_speed = _extract_flight_speed(tags, "X")
    flight_y_speed = _extract_flight_speed(tags, "Y")
    flight_z_speed = _extract_flight_speed(tags, "Z")
    gimbal_pitch = _extract_gimbal_pitch(tags)
    gimbal_roll = _extract_gimbal_roll(tags)
    product_name = _extract_first_text(
        tags,
        (
            "Product Name",
            "XMP DJI:ProductName",
            "ProductName",
        ),
    )
    unique_camera_model = _extract_first_text(
        tags,
        (
            "Unique Camera Model",
            "EXIF UniqueCameraModel",
            "UniqueCameraModel",
        ),
    )
    captured_at = _extract_capture_time(tags)

    if (
        (flight_pitch is not None and abs(float(flight_pitch)) > 25.0)
        or (flight_roll is not None and abs(float(flight_roll)) > 25.0)
    ):
        sys.stderr.write(
            "WARN: Flight Pitch Degree or Flight Roll Degree exceeds +/-25 deg; "
            f"Gimbal may not be level: {path} "
            f"flight_pitch={flight_pitch} flight_roll={flight_roll}\n"
        )

    if gimbal_pitch is not None and abs(float(gimbal_pitch) - (-90.0)) > 1e-6:
        sys.stderr.write(
            "WARN: Gimbal Pitch Degree is not -90 deg; "
            f"Gimbal may not be level: {path} gimbal_pitch={gimbal_pitch}\n"
        )

    if gimbal_roll is not None and abs(float(gimbal_roll)) > 1e-6:
        sys.stderr.write(
            "WARN: Gimbal Roll Degree is not 0 deg; "
            f"Gimbal may not be level: {path} gimbal_roll={gimbal_roll}\n"
        )

    hfov_override = None if options is None else getattr(options, "hfov_deg_override", None)
    hfov = float(hfov_override) if hfov_override is not None else _extract_fov(tags, w, h)

    # If we couldn't parse required fields, fail (caller will report details)
    if lat is None or lon is None or alt is None or yaw is None:
        return None

    if hfov is None:
        # As a last resort, try diagonal FOV from a tag and convert.
        dfov = None
        for k in ("XMP Camera:DiagonalFOV", "XMP DJI:DiagonalFOV", "Image DiagonalFOV"):
            if k in tags:
                dfov = _try_parse_float(tags.get(k))
                if dfov:
                    break
        if dfov is not None and dfov > 0:
            hfov = _approx_hfov_from_dfov(float(dfov), w, h)
        else:
            return None

    return PhotoMeta(
        path=path,
        w=w,
        h=h,
        lat_deg=float(lat),
        lon_deg=float(lon),
        alt_m=float(alt),
        absolute_alt_m=float(absolute_alt) if absolute_alt is not None else None,
        yaw_deg=float(yaw) % 360.0,
        flight_yaw_deg=float(flight_yaw) % 360.0 if flight_yaw is not None else None,
        gimbal_yaw_deg=float(gimbal_yaw) % 360.0 if gimbal_yaw is not None else None,
        hfov_deg=float(hfov),
        pitch_deg=float(pitch) if pitch is not None else -90.0,
        flight_pitch_deg=float(flight_pitch) if flight_pitch is not None else None,
        gimbal_pitch_deg=float(gimbal_pitch) if gimbal_pitch is not None else None,
        gimbal_roll_deg=float(gimbal_roll) if gimbal_roll is not None else None,
        flight_roll_deg=float(flight_roll) if flight_roll is not None else None,
        flight_x_speed_mps=float(flight_x_speed) if flight_x_speed is not None else None,
        flight_y_speed_mps=float(flight_y_speed) if flight_y_speed is not None else None,
        flight_z_speed_mps=float(flight_z_speed) if flight_z_speed is not None else None,
        product_name=product_name,
        unique_camera_model=unique_camera_model,
        captured_at=captured_at,
    )


def _list_images(in_dir: str) -> List[str]:
    paths: List[str] = []
    for p in glob.glob(os.path.join(in_dir, "**", "*"), recursive=True):
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
            paths.append(p)
    paths.sort()
    return paths


def _meters_per_deg(lat_deg: float) -> Tuple[float, float]:
    # Approx WGS84 meters per degree.
    lat = math.radians(lat_deg)
    m_per_deg_lat = (
        111132.954
        - 559.822 * math.cos(2 * lat)
        + 1.175 * math.cos(4 * lat)
        - 0.0023 * math.cos(6 * lat)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat)
        - 93.5 * math.cos(3 * lat)
        + 0.118 * math.cos(5 * lat)
    )
    return m_per_deg_lat, m_per_deg_lon


def _yaw_to_image_axes(yaw_deg: float, options: Optional[RenderOptions] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (right_world, down_world) unit vectors in world meters.

    World axes: +X east, +Y north.
    yaw_deg is degrees clockwise from true north of the camera/image TOP direction.

    By default we use +yaw. Use --yaw-invert to switch to -yaw (legacy behavior).
    """
    off = 0.0 if options is None else float(getattr(options, "yaw_offset_deg", 0.0))
    inv = False if options is None else bool(getattr(options, "yaw_invert", False))
    sign = -1.0 if inv else 1.0
    a = math.radians((sign * (float(yaw_deg) + off)) % 360.0)

    top_world = np.array([math.sin(a), math.cos(a)], dtype=np.float64)
    right_world = np.array([math.cos(a), -math.sin(a)], dtype=np.float64)
    down_world = -top_world
    return right_world, down_world


def _yaw_to_top_world(yaw_deg: float, options: Optional[RenderOptions] = None) -> np.ndarray:
    """Unit vector (east,north) of image top direction in world."""
    off = 0.0 if options is None else float(getattr(options, "yaw_offset_deg", 0.0))
    inv = False if options is None else bool(getattr(options, "yaw_invert", False))
    sign = -1.0 if inv else 1.0
    a = math.radians((sign * (float(yaw_deg) + off)) % 360.0)
    return np.array([math.sin(a), math.cos(a)], dtype=np.float64)


def _compute_mpp(meta: PhotoMeta, options: Optional[RenderOptions] = None) -> float:
    # meters-per-pixel at target plane for horizontal axis
    if options is None:
        alt = float(meta.alt_m)
    else:
        alt = _effective_alt_m(meta, options)
    ground_w_m = 2.0 * alt * math.tan(math.radians(meta.hfov_deg) / 2.0)
    return ground_w_m / float(meta.w)


def _project_corners(meta: PhotoMeta, origin_lat: float, origin_lon: float, options: Optional[RenderOptions] = None) -> np.ndarray:
    # Returns 4x2 array in meters relative to origin: [[x_east, y_north], ...] for image corners
    m_per_deg_lat, m_per_deg_lon = _meters_per_deg(origin_lat)
    x0 = (meta.lon_deg - origin_lon) * m_per_deg_lon
    y0 = (meta.lat_deg - origin_lat) * m_per_deg_lat

    # If camera is tilted, shift ground intersection of image center forward.
    # Approx: forward displacement = alt * tan(tilt)
    # Apply this shift only when use_pitch is set
    if options is not None and bool(getattr(options, "use_pitch", False)):
        alt_eff = float(meta.alt_m) + float(getattr(options, "alt_correction_m", 0.0))
        tilt_deg = _tilt_from_nadir_deg(getattr(meta, "pitch_deg", -90.0))
        forward = alt_eff * math.tan(math.radians(tilt_deg))
        top_world = _yaw_to_top_world(meta.yaw_deg, options)
        # image top is forward in world, so center shifts toward top_world
        x0 += float(top_world[0]) * forward
        y0 += float(top_world[1]) * forward

    mpp = _compute_mpp(meta, options)
    right_world, down_world = _yaw_to_image_axes(meta.yaw_deg, options)

    # pixel coords at corners relative to center
    cx = meta.w / 2.0
    cy = meta.h / 2.0
    corners_px = np.array(
        [
            [0.0 - cx, 0.0 - cy],
            [meta.w - cx, 0.0 - cy],
            [meta.w - cx, meta.h - cy],
            [0.0 - cx, meta.h - cy],
        ],
        dtype=np.float64,
    )

    # Map pixel delta to meters in world plane
    # world_delta = (dx * mpp) * right_world + (dy * mpp) * down_world
    deltas = (
        np.outer(corners_px[:, 0] * mpp, right_world)
        + np.outer(corners_px[:, 1] * mpp, down_world)
    )
    corners_world = deltas + np.array([x0, y0], dtype=np.float64)
    return corners_world


def _compute_canvas(
    photos: List[PhotoMeta],
    options: Optional[RenderOptions] = None,
) -> Tuple[float, float, float, float, float, float, float]:
    # Determine origin at mean lat/lon to keep meters-per-degree stable.
    origin_lat = float(np.mean([p.lat_deg for p in photos]))
    origin_lon = float(np.mean([p.lon_deg for p in photos]))

    all_pts = []
    for p in photos:
        c = _project_corners(p, origin_lat, origin_lon, options)
        all_pts.append(c)
    pts = np.vstack(all_pts)

    min_x = float(np.min(pts[:, 0]))
    max_x = float(np.max(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_y = float(np.max(pts[:, 1]))

    # Choose output meters-per-pixel from lowest effective altitude (smallest mpp)
    if options is None:
        mpp = float(min(_compute_mpp(p) for p in photos))
    else:
        mpp = float(min(_compute_mpp(p, options) for p in photos))
    return origin_lat, origin_lon, min_x, min_y, max_x, max_y, mpp


def _world_to_canvas(x: np.ndarray, y: np.ndarray, min_x: float, max_y: float, mpp: float) -> Tuple[np.ndarray, np.ndarray]:
    # Canvas coordinates: x right, y down
    u = (x - min_x) / mpp
    v = (max_y - y) / mpp
    return u, v


def _undistort_rgba(im: Image.Image, k1: float, k2: float, k3: float) -> Image.Image:
    """Simple radial undistortion with 3-parameter model.

    This is a lightweight approximation:
      r_d = r_u * (1 + k1*r_u^2 + k2*r_u^4 + k3*r_u^6)
    We compute inverse mapping (dst->src) to sample source.

    k1/k2/k3 units are normalized to image half-diagonal (r in [0,1]).
    """
    if all(abs(float(k)) < 1e-12 for k in (k1, k2, k3)):
        return im

    im = im.convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    # normalize by half diagonal
    rn = math.sqrt(cx * cx + cy * cy)
    if rn <= 0:
        return im

    yy, xx = np.indices((h, w), dtype=np.float64)
    x = (xx - cx) / rn
    y = (yy - cy) / rn
    r2 = x * x + y * y

    # Inverse mapping: we want src coords for each dst pixel.
    # Approximate inverse by dividing by the radial polynomial.
    scale = 1.0 + float(k1) * r2 + float(k2) * (r2 * r2) + float(k3) * (r2 * r2 * r2)
    scale = np.where(scale == 0, 1.0, scale)
    xu = x / scale
    yu = y / scale

    xs = xu * rn + cx
    ys = yu * rn + cy

    # Bilinear sampling
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)

    wa = wa[..., None]
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]

    out = (
        arr[y0c, x0c].astype(np.float64) * wa
        + arr[y0c, x1c].astype(np.float64) * wb
        + arr[y1c, x0c].astype(np.float64) * wc
        + arr[y1c, x1c].astype(np.float64) * wd
    )

    # Outside original bounds: make transparent
    mask = (xs < 0) | (xs > (w - 1)) | (ys < 0) | (ys > (h - 1))
    out[mask] = np.array([0, 0, 0, 0], dtype=np.float64)

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out_u8, mode="RGBA")


def _alpha_composite(base: Image.Image, over: Image.Image) -> Image.Image:
    # base and over are RGBA
    return Image.alpha_composite(base, over)


def _apply_opacity_rgba(im: Image.Image, opacity_pct: float) -> Image.Image:
    """Scale alpha channel by opacity_pct (0-100)."""
    op = float(opacity_pct)
    if op >= 100.0:
        return im
    if op <= 0.0:
        # fully transparent
        return Image.new("RGBA", im.size, (0, 0, 0, 0))

    arr = np.array(im, dtype=np.uint8)
    a = arr[..., 3].astype(np.float32)
    a = a * (op / 100.0)
    arr[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _inspect_exif_for_file(path: str) -> str:
    """Return a single-line report for one file."""
    try:
        tags = _read_exif_tags(path)
    except Exception as e:
        return f"{path}: EXIF unreadable ({e})"

    # Evaluate presence and parsability of the fields we actually use
    lat, lon = _extract_lat_lon(tags)
    alt = _extract_alt(tags)
    yaw = _extract_yaw(tags)
    pitch = _extract_gimbal_pitch(tags)
    roll = _extract_gimbal_roll(tags)

    # Collect raw yaw/pitch/roll candidates
    yaw_raw: List[str] = []
    pitch_raw: List[str] = []
    roll_raw: List[str] = []
    for k in INSPECT_KEYS.get("flight_yaw", ()):
        kk = _find_tag_key(tags, (k,))
        if kk is not None:
            yaw_raw.append(f"{kk}={_short(tags.get(kk))}")
    for k in INSPECT_KEYS.get("gimbal_yaw", ()):
        kk = _find_tag_key(tags, (k,))
        if kk is not None:
            yaw_raw.append(f"{kk}={_short(tags.get(kk))}")
    for k in INSPECT_KEYS.get("gimbal_pitch", ()):
        kk = _find_tag_key(tags, (k,))
        if kk is not None:
            pitch_raw.append(f"{kk}={_short(tags.get(kk))}")
    for k in INSPECT_KEYS.get("gimbal_roll", ()):
        kk = _find_tag_key(tags, (k,))
        if kk is not None:
            roll_raw.append(f"{kk}={_short(tags.get(kk))}")

    w = h = None
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        pass

    hfov = None
    if w is not None and h is not None:
        try:
            hfov = _extract_fov(tags, w, h)
            if hfov is None:
                # try diagonal->horizontal path similarly to _load_photo_meta
                dfov = None
                for k in ("XMP Camera:DiagonalFOV", "XMP DJI:DiagonalFOV", "Image DiagonalFOV"):
                    kk = _find_tag_key(tags, (k,))
                    if kk is not None:
                        dfov = _try_parse_float(tags.get(kk))
                        if dfov:
                            break
                if dfov is not None and dfov > 0:
                    hfov = _approx_hfov_from_dfov(float(dfov), w, h)
        except Exception:
            hfov = None

    def yn(ok: bool) -> str:
        return "OK" if ok else "--"

    # Also show raw-key presence for debugging
    present = {name: [k for k in keys if _find_tag_key(tags, (k,)) is not None] for name, keys in INSPECT_KEYS.items()}

    warn_parts: List[str] = []
    # gimbal pitch should be near -90 ± 2
    if pitch is not None and abs(float(pitch) - (-90.0)) > 2.0:
        warn_parts.append(f"gimbal_pitch={float(pitch):.2f}")
    # gimbal roll should be 0
    if roll is not None and abs(float(roll)) > 1e-6:
        warn_parts.append(f"gimbal_roll={float(roll):.2f}")

    warn_txt = ""
    if warn_parts:
        warn_txt = " WARN(" + ",".join(warn_parts) + ")"

    raw_note = ""
    if yaw is None and yaw_raw:
        raw_note += " yaw_raw=[" + "; ".join(yaw_raw[:2]) + "]"
    if pitch is None and pitch_raw:
        raw_note += " pitch_raw=[" + "; ".join(pitch_raw[:1]) + "]"
    if roll is None and roll_raw:
        raw_note += " roll_raw=[" + "; ".join(roll_raw[:1]) + "]"

    return (
        f"{path}: "
        f"lat={yn(lat is not None)} lon={yn(lon is not None)} "
        f"alt={yn(alt is not None)} yaw={yn(yaw is not None)} "
        f"pitch={yn(pitch is not None)} roll={yn(roll is not None)} fov={yn(hfov is not None)}"
        f"{warn_txt} "
        f"keys={{"
        f"lat:{present['lat'][:2]} lon:{present['lon'][:2]} alt:{present['alt'][:2]} "
        f"flight_yaw:{present['flight_yaw'][:1]} gimbal_yaw:{present['gimbal_yaw'][:1]} "
        f"gimbal_pitch:{present['gimbal_pitch'][:1]} gimbal_roll:{present['gimbal_roll'][:1]} fov:{present['fov'][:2]}"
        f"}}"
        f"{raw_note}"
    )


def inspect_directory(in_dir: str) -> None:
    paths = _list_images(in_dir)
    if not paths:
        raise SystemExit(f"No images found in: {in_dir}")
    sys.stderr.write(f"Inspecting {len(paths)} files in: {in_dir}\n")

    pitches: List[str] = []
    rolls: List[str] = []

    for p in paths:
        # Existing per-file report
        sys.stderr.write(_inspect_exif_for_file(p) + "\n")

        # Collect numeric values for summary
        try:
            tags = _read_exif_tags(p)
            gp = _extract_gimbal_pitch(tags)
            gr = _extract_gimbal_roll(tags)
            pitches.append(f"{float(gp):.2f}" if gp is not None else "--")
            rolls.append(f"{float(gr):.2f}" if gr is not None else "--")
        except Exception:
            pitches.append("--")
            rolls.append("--")

    # One-line summaries
    sys.stderr.write("gimbal_pitch: " + " ".join(pitches) + "\n")
    sys.stderr.write("gimbal_roll:  " + " ".join(rolls) + "\n")


def _alpha_blend_rgba_over_rgb_inplace(base_rgb: Image.Image, over_rgba: Image.Image, dest: Tuple[int, int]) -> None:
    """Alpha-blend an RGBA ROI over an RGB base image in-place.

    base_rgb: full canvas RGB image
    over_rgba: ROI-sized RGBA image
    dest: (x,y) top-left in base
    """
    if base_rgb.mode != "RGB":
        raise ValueError("base_rgb must be RGB")
    if over_rgba.mode != "RGBA":
        over_rgba = over_rgba.convert("RGBA")

    x, y = int(dest[0]), int(dest[1])
    if over_rgba.width <= 0 or over_rgba.height <= 0:
        return

    # Clip ROI to base bounds
    bx0 = max(0, x)
    by0 = max(0, y)
    bx1 = min(base_rgb.width, x + over_rgba.width)
    by1 = min(base_rgb.height, y + over_rgba.height)
    if bx1 <= bx0 or by1 <= by0:
        return

    ox0 = bx0 - x
    oy0 = by0 - y
    ox1 = ox0 + (bx1 - bx0)
    oy1 = oy0 + (by1 - by0)

    base_crop = base_rgb.crop((bx0, by0, bx1, by1))
    over_crop = over_rgba.crop((ox0, oy0, ox1, oy1))

    b = np.asarray(base_crop, dtype=np.float32)  # (h,w,3)
    o = np.asarray(over_crop, dtype=np.float32)  # (h,w,4)

    a = (o[..., 3:4] / 255.0)
    out = o[..., :3] * a + b * (1.0 - a)

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    base_rgb.paste(Image.fromarray(out_u8, mode="RGB"), (bx0, by0))


def _alpha_blend_rgba_over_rgba_inplace(base_rgba: Image.Image, over_rgba: Image.Image, dest: Tuple[int, int]) -> None:
    """Alpha-composite an RGBA ROI over an RGBA base image in-place."""
    if base_rgba.mode != "RGBA":
        raise ValueError("base_rgba must be RGBA")
    if over_rgba.mode != "RGBA":
        over_rgba = over_rgba.convert("RGBA")

    x, y = int(dest[0]), int(dest[1])
    if over_rgba.width <= 0 or over_rgba.height <= 0:
        return

    bx0 = max(0, x)
    by0 = max(0, y)
    bx1 = min(base_rgba.width, x + over_rgba.width)
    by1 = min(base_rgba.height, y + over_rgba.height)
    if bx1 <= bx0 or by1 <= by0:
        return

    ox0 = bx0 - x
    oy0 = by0 - y
    ox1 = ox0 + (bx1 - bx0)
    oy1 = oy0 + (by1 - by0)

    base_crop = base_rgba.crop((bx0, by0, bx1, by1))
    over_crop = over_rgba.crop((ox0, oy0, ox1, oy1))
    composited = Image.alpha_composite(base_crop, over_crop)
    base_rgba.paste(composited, (bx0, by0))


def _photo_center_canvas_xy(
    meta: PhotoMeta,
    origin_lat: float,
    origin_lon: float,
    min_x: float,
    max_y: float,
    mpp_out: float,
    options: RenderOptions,
) -> Tuple[float, float]:
    """Project photo center to canvas coordinates."""
    m_per_deg_lat, m_per_deg_lon = _meters_per_deg(origin_lat)
    x0 = (meta.lon_deg - origin_lon) * m_per_deg_lon
    y0 = (meta.lat_deg - origin_lat) * m_per_deg_lat

    if bool(getattr(options, "use_pitch", False)):
        alt_eff = float(meta.alt_m) + float(getattr(options, "alt_correction_m", 0.0))
        tilt_deg = _tilt_from_nadir_deg(getattr(meta, "pitch_deg", -90.0))
        forward = alt_eff * math.tan(math.radians(tilt_deg))
        top_world = _yaw_to_top_world(meta.yaw_deg, options)
        x0 += float(top_world[0]) * forward
        y0 += float(top_world[1]) * forward

    u, v = _world_to_canvas(
        np.array([x0], dtype=np.float64),
        np.array([y0], dtype=np.float64),
        min_x=min_x,
        max_y=max_y,
        mpp=mpp_out,
    )
    return float(u[0]), float(v[0])


def _composite_nearest_center_rgba_inplace(
    base_rgba: Image.Image,
    best_dist2: np.ndarray,
    over_rgba: Image.Image,
    dest: Tuple[int, int],
    center_u: float,
    center_v: float,
) -> None:
    """Per-pixel winner-takes-front by nearest image center distance."""
    if base_rgba.mode != "RGBA":
        raise ValueError("base_rgba must be RGBA")
    if over_rgba.mode != "RGBA":
        over_rgba = over_rgba.convert("RGBA")

    x, y = int(dest[0]), int(dest[1])
    if over_rgba.width <= 0 or over_rgba.height <= 0:
        return

    bx0 = max(0, x)
    by0 = max(0, y)
    bx1 = min(base_rgba.width, x + over_rgba.width)
    by1 = min(base_rgba.height, y + over_rgba.height)
    if bx1 <= bx0 or by1 <= by0:
        return

    ox0 = bx0 - x
    oy0 = by0 - y
    ox1 = ox0 + (bx1 - bx0)
    oy1 = oy0 + (by1 - by0)

    over_crop = np.asarray(over_rgba.crop((ox0, oy0, ox1, oy1)), dtype=np.uint8)
    alpha_mask = over_crop[..., 3] > 0
    if not np.any(alpha_mask):
        return

    h = by1 - by0
    w = bx1 - bx0
    xs = np.arange(w, dtype=np.float32) + np.float32(bx0)
    ys = np.arange(h, dtype=np.float32) + np.float32(by0)
    dx2 = (xs - np.float32(center_u)) ** 2
    dy2 = (ys - np.float32(center_v)) ** 2
    d2 = dy2[:, None] + dx2[None, :]

    dist_crop = best_dist2[by0:by1, bx0:bx1]
    update_mask = alpha_mask & (d2 < dist_crop)
    if not np.any(update_mask):
        return

    # np.asarray(PIL.Image) can be read-only; use np.array(copy=True) for writable buffer.
    base_crop = np.array(base_rgba.crop((bx0, by0, bx1, by1)), dtype=np.uint8, copy=True)
    base_crop[update_mask] = over_crop[update_mask]
    dist_crop[update_mask] = d2[update_mask]
    base_rgba.paste(Image.fromarray(base_crop, mode="RGBA"), (bx0, by0))


def _save_image_with_options(
    img: Image.Image,
    out_target: Union[str, BinaryIO],
    options: RenderOptions,
    out_path_hint: Optional[str] = None,
) -> None:
    """Save image with format-specific options based on file extension."""
    resolved_path = out_path_hint if out_path_hint is not None else str(out_target)
    ext = os.path.splitext(resolved_path)[1].lower()

    if ext in (".jpg", ".jpeg"):
        q = int(getattr(options, "jpg_quality", 95))
        q = max(1, min(95, q))
        # JPEG has no alpha; flatten transparent pixels over white.
        if img.mode != "RGB":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                bg.paste(img, mask=img.split()[-1])
            else:
                bg.paste(img.convert("RGB"))
            img_jpg = bg
        else:
            img_jpg = img
        img_jpg.save(out_target, format="JPEG", quality=q, optimize=True, subsampling=0)
        return

    if ext == ".png":
        cl = int(getattr(options, "png_compress_level", 6))
        cl = max(0, min(9, cl))
        img.save(out_target, format="PNG", compress_level=cl)
        return

    # Fallback: rely on Pillow defaults (e.g., tif/tiff)
    img.save(out_target)


def _cli_save_progress_printer(
    state: SaveProgressState,
    stop_event: threading.Event,
    widths: List[int],
    interval_sec: float = 0.5,
) -> None:
    while not stop_event.wait(interval_sec):
        line = _format_save_progress_text(state)
        widths[0] = max(widths[0], len(line))
        sys.stderr.write("\r" + line.ljust(widths[0]))
        sys.stderr.flush()


def _ensure_output_dimension_limit(out_path: str, canvas_w: int, canvas_h: int) -> None:
    """Abort when JPEG/TIFF output exceeds the practical 64k-side limit."""
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".tif", ".tiff"):
        return

    if canvas_w > MAX_JPG_TIF_DIM_PX or canvas_h > MAX_JPG_TIF_DIM_PX:
        raise SystemExit(
            "WARNING: The computed canvas size is "
            f"{canvas_w}x{canvas_h}, and one side exceeds 64K pixels. "
            "JPEG/TIFF output may fail or be unsupported at this size. "
            "Please use a .png extension for --out."
        )


def _ensure_save_memory_headroom(
    canvas_w: int,
    canvas_h: int,
    out_path: str,
    metas: List[PhotoMeta],
    options: RenderOptions,
    mpp_out: float,
) -> None:
    use_crop_opt = bool(getattr(options, "crop_optimize", False))
    canvas_bytes = _estimated_canvas_memory_bytes(canvas_w, canvas_h, out_path, use_crop_opt)
    peak_photo_bytes = 0 if not metas else max(
        _estimate_photo_peak_bytes(meta, canvas_w, canvas_h, mpp_out, options) for meta in metas
    )
    _ensure_memory_headroom(
        canvas_bytes + peak_photo_bytes,
        context=f"allocating {canvas_w}x{canvas_h} save buffers",
    )


def mosaic(in_dir: str, out_path: str, options: RenderOptions) -> None:
    # Preflight inspection removed; only --inspect-only in main() will run inspect_directory(in_dir)
    paths = _list_images(in_dir)
    if not paths:
        raise SystemExit(f"No images found in: {in_dir}")

    metas: List[PhotoMeta] = []
    skipped: List[str] = []
    for p in paths:
        try:
            m = _load_photo_meta(p, options)
        except Exception:
            m = None
        if m is None:
            # Re-read tags for diagnostics (best-effort)
            reason_parts: List[str] = []
            try:
                tags = _read_exif_tags(p)
                lat, lon = _extract_lat_lon(tags)
                alt = _extract_alt(tags)
                yaw = _extract_yaw(tags, options)
                with Image.open(p) as im:
                    w, h = im.size
                hfov = _extract_fov(tags, w, h)
                if lat is None:
                    reason_parts.append("lat")
                if lon is None:
                    reason_parts.append("lon")
                if alt is None:
                    reason_parts.append("alt")
                if yaw is None:
                    reason_parts.append("yaw")
                if hfov is None:
                    reason_parts.append("fov")
                pitch = _extract_gimbal_pitch(tags)
                if pitch is not None and abs(float(pitch) - (-90.0)) > 1.0:
                    reason_parts.append(f"pitch={pitch}")
            except Exception:
                reason_parts.append("unreadable")

            skipped.append(p + (" (missing: " + ",".join(reason_parts) + ")" if reason_parts else ""))
            continue
        metas.append(m)

    if not metas:
        raise SystemExit("No usable images with required EXIF (lat/lon/alt/yaw/fov).")

    origin_lat, origin_lon, min_x, min_y, max_x, max_y, mpp = _compute_canvas(metas, options)

    canvas_w = int(math.ceil((max_x - min_x) / mpp))
    canvas_h = int(math.ceil((max_y - min_y) / mpp))
    _ensure_output_dimension_limit(out_path, canvas_w, canvas_h)
    _ensure_save_memory_headroom(canvas_w, canvas_h, out_path, metas, options, mpp)

    # Keep base as RGBA with transparent background (no-source regions remain transparent)
    base = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Paint higher-altitude first, then lower-altitude last (sharper over)
    metas_sorted = sorted(metas, key=lambda m: _effective_alt_m(m, options), reverse=True)
    use_crop_opt = bool(getattr(options, "crop_optimize", False))
    best_dist2: Optional[np.ndarray] = None
    if use_crop_opt:
        best_dist2 = np.full((canvas_h, canvas_w), np.inf, dtype=np.float32)

    total = len(metas_sorted)
    bar_len = 40

    for idx, meta in enumerate(metas_sorted):
        # Print brief progress header (overwritten by bar update)
        fname = os.path.basename(meta.path)
        sys.stderr.write(f"Processing {idx+1}/{total}: {fname}\n")
        sys.stderr.flush()

        _ensure_memory_headroom(
            _estimate_photo_peak_bytes(meta, canvas_w, canvas_h, mpp, options),
            context=f"processing {fname}",
            margin_bytes=MEMORY_SAFETY_MARGIN_BYTES // 2,
        )

        warped, offset = _warp_photo_to_canvas(
            meta,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            min_x=min_x,
            max_y=max_y,
            mpp_out=mpp,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            options=options,
        )
        if use_crop_opt and best_dist2 is not None:
            cu, cv = _photo_center_canvas_xy(
                meta,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                min_x=min_x,
                max_y=max_y,
                mpp_out=mpp,
                options=options,
            )
            _composite_nearest_center_rgba_inplace(base, best_dist2, warped, offset, cu, cv)
        else:
            _alpha_blend_rgba_over_rgba_inplace(base, warped, offset)
        # Help GC for huge datasets
        del warped

        # Update inline textual progress bar
        done = idx + 1
        pct = int((done / total) * 100)
        filled = int((done / total) * bar_len)
        bar = "[" + ("#" * filled) + ("-" * (bar_len - filled)) + "]"
        sys.stderr.write(f"\r{bar} {pct:3d}% ({done}/{total}) {fname}")
        sys.stderr.flush()

    # Finish progress line
    sys.stderr.write("\n")

    # Save image (PNG keeps alpha; JPEG is flattened to white)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Notify user that save is starting (can be slow) and measure duration
    sys.stderr.write(f"Saving to {out_path} ... this may take a while\n")
    sys.stderr.flush()
    t_save0 = time.perf_counter()
    save_state = SaveProgressState(start_time=t_save0)
    save_stop = threading.Event()
    cli_widths = [0]
    save_monitor = threading.Thread(
        target=_cli_save_progress_printer,
        args=(save_state, save_stop, cli_widths, 0.5),
        daemon=True,
    )
    save_monitor.start()
    try:
        _save_image_job(base, out_path, options, save_state)
    except Exception:
        partial_save_line = _format_save_progress_text(save_state)
        cli_widths[0] = max(cli_widths[0], len(partial_save_line))
        sys.stderr.write("\r" + partial_save_line.ljust(cli_widths[0]) + "\n")
        sys.stderr.flush()
        raise
    finally:
        save_stop.set()
        save_monitor.join()
    t_save = time.perf_counter() - t_save0
    final_save_line = _format_save_progress_text(save_state)
    cli_widths[0] = max(cli_widths[0], len(final_save_line))
    sys.stderr.write("\r" + final_save_line.ljust(cli_widths[0]) + "\n")
    sys.stderr.flush()

    # After saving (headless), report full path, resolution and file size
    try:
        saved_path = os.path.abspath(out_path)
        w, h = base.size
        size_bytes = os.path.getsize(saved_path)
        def _hr(n):
            for unit in ("B", "KB", "MB", "GB"):
                if n < 1024.0:
                    return f"{n:.1f}{unit}"
                n /= 1024.0
            return f"{n:.1f}TB"
        size_hr = _hr(size_bytes)
        # Include JPEG/PNG quality info if available
        ext = os.path.splitext(saved_path)[1].lower()
        qinfo = ""
        if ext in (".jpg", ".jpeg"):
            qinfo = f"\nJPEG quality: {int(getattr(options, 'jpg_quality', 95))}"
        elif ext == ".png":
            qinfo = f"\nPNG compress level: {int(getattr(options, 'png_compress_level', 6))}"
        msg = f"Saved to: {saved_path}\nResolution: {w}x{h}\nFile size: {size_hr} ({size_bytes} bytes){qinfo}\nSave time: {t_save:.2f} sec"
    except Exception as e:
        msg = f"Saved to: {os.path.abspath(out_path)}\n(Additional info unavailable: {e})"
    sys.stderr.write(msg + "\n")

    if skipped:
        sys.stderr.write("Skipped files (missing required metadata):\n")
        for p in skipped:
            sys.stderr.write(f"  {p}\n")


def _warp_photo_to_canvas(
    meta: PhotoMeta,
    origin_lat: float,
    origin_lon: float,
    min_x: float,
    max_y: float,
    mpp_out: float,
    canvas_w: int,
    canvas_h: int,
    options: RenderOptions,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Warp one photo into the global canvas.

    When options.roi_warp is True, returns a small RGBA ROI image and its paste offset.
    Otherwise returns a full-canvas RGBA image and offset (0,0).
    """
    # Position of photo center in world meters
    m_per_deg_lat, m_per_deg_lon = _meters_per_deg(origin_lat)
    x0 = (meta.lon_deg - origin_lon) * m_per_deg_lon
    y0 = (meta.lat_deg - origin_lat) * m_per_deg_lat

    # Tilt center shift (same as _project_corners)
    # Apply this shift only when use_pitch is set
    if bool(getattr(options, "use_pitch", False)):
        alt_eff = float(meta.alt_m) + float(getattr(options, "alt_correction_m", 0.0))
        tilt_deg = _tilt_from_nadir_deg(getattr(meta, "pitch_deg", -90.0))
        forward = alt_eff * math.tan(math.radians(tilt_deg))
        top_world = _yaw_to_top_world(meta.yaw_deg, options)
        x0 += float(top_world[0]) * forward
        y0 += float(top_world[1]) * forward

    mpp_in = _compute_mpp(meta, options)
    right_world, down_world = _yaw_to_image_axes(meta.yaw_deg, options)

    cx = meta.w / 2.0
    cy = meta.h / 2.0

    def src_to_world_xy(xp: float, yp: float) -> Tuple[float, float]:
        dx = (xp - cx) * mpp_in
        dy = (yp - cy) * mpp_in
        wpt = np.array([x0, y0], dtype=np.float64) + dx * right_world + dy * down_world
        return float(wpt[0]), float(wpt[1])

    # Project 4 corners to canvas to compute ROI
    use_roi = bool(getattr(options, "roi_warp", True))
    margin = int(getattr(options, "roi_margin_px", 8))
    corners_src = [
        (0.0, 0.0),
        (meta.w - 1.0, 0.0),
        (meta.w - 1.0, meta.h - 1.0),
        (0.0, meta.h - 1.0),
    ]

    uv = []
    for xp, yp in corners_src:
        wx, wy = src_to_world_xy(xp, yp)
        u, v = _world_to_canvas(
            np.array([wx], dtype=np.float64),
            np.array([wy], dtype=np.float64),
            min_x=min_x,
            max_y=max_y,
            mpp=mpp_out,
        )
        uv.append((float(u[0]), float(v[0])))

    umin = min(p[0] for p in uv)
    umax = max(p[0] for p in uv)
    vmin = min(p[1] for p in uv)
    vmax = max(p[1] for p in uv)

    if use_roi:
        x1 = max(0, int(math.floor(umin)) - margin)
        y1 = max(0, int(math.floor(vmin)) - margin)
        x2 = min(canvas_w, int(math.ceil(umax)) + margin)
        y2 = min(canvas_h, int(math.ceil(vmax)) + margin)
        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)
        out_size = (roi_w, roi_h)
        offset = (x1, y1)
    else:
        out_size = (canvas_w, canvas_h)
        offset = (0, 0)

    # Use 3 non-collinear points to solve affine (src->dst)
    src3 = np.array([[0.0, 0.0], [meta.w - 1.0, 0.0], [0.0, meta.h - 1.0]], dtype=np.float64)

    dst3 = []
    for i in range(3):
        wx, wy = src_to_world_xy(float(src3[i, 0]), float(src3[i, 1]))
        u, v = _world_to_canvas(
            np.array([wx], dtype=np.float64),
            np.array([wy], dtype=np.float64),
            min_x=min_x,
            max_y=max_y,
            mpp=mpp_out,
        )
        du = float(u[0]) - float(offset[0])
        dv = float(v[0]) - float(offset[1])
        dst3.append([du, dv])
    dst3 = np.array(dst3, dtype=np.float64)

    def affine_from_points(p_src: np.ndarray, p_dst: np.ndarray) -> np.ndarray:
        x = p_src[:, 0]
        y = p_src[:, 1]
        u = p_dst[:, 0]
        v = p_dst[:, 1]
        A = np.array(
            [
                [x[0], y[0], 1, 0, 0, 0],
                [0, 0, 0, x[0], y[0], 1],
                [x[1], y[1], 1, 0, 0, 0],
                [0, 0, 0, x[1], y[1], 1],
                [x[2], y[2], 1, 0, 0, 0],
                [0, 0, 0, x[2], y[2], 1],
            ],
            dtype=np.float64,
        )
        b = np.array([u[0], v[0], u[1], v[1], u[2], v[2]], dtype=np.float64)
        params = np.linalg.solve(A, b)
        return np.array(
            [[params[0], params[1], params[2]], [params[3], params[4], params[5]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    M_src_to_dst = affine_from_points(src3, dst3)
    M_dst_to_src = np.linalg.inv(M_src_to_dst)
    a, b, c = float(M_dst_to_src[0, 0]), float(M_dst_to_src[0, 1]), float(M_dst_to_src[0, 2])
    d, e, f = float(M_dst_to_src[1, 0]), float(M_dst_to_src[1, 1]), float(M_dst_to_src[1, 2])
    affine6 = (a, b, c, d, e, f)

    with Image.open(meta.path) as im:
        im = im.convert("RGBA")
        if options.undistort:
            im = _undistort_rgba(im, options.k1, options.k2, options.k3)
        warped = im.transform(
            out_size,
            Image.AFFINE,
            data=affine6,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0, 0),
        )

    # Apply global opacity
    warped = _apply_opacity_rgba(warped, float(getattr(options, "opacity_pct", 100.0)))

    return warped, offset


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mosaic drone images using EXIF lat/lon/alt/yaw/fov")
    ap.add_argument("--in", dest="in_dir", required=True, help="input directory")
    ap.add_argument("--out", dest="out", required=True, help="output image path")
    ap.add_argument(
        "--opacity",
        type=float,
        default=100.0,
        help="global layer opacity in percent (0-100). Default: 100",
    )
    ap.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality (1-95). Used only when --out ends with .jpg/.jpeg (default: 95)",
    )
    ap.add_argument(
        "--png-compress-level",
        type=int,
        default=6,
        help="PNG compress level (0-9). Used only when --out ends with .png (default: 6)",
    )
    ap.add_argument(
        "--alt-correction",
        type=float,
        default=0.0,
        help="altitude correction in meters; effective altitude is Relative Altitude + alt_correction",
    )
    ap.add_argument(
        "--hfov-deg",
        "--fov",
        dest="hfov_deg",
        type=float,
        default=None,
        help="override horizontal FOV in degrees for all images (default: read/derive from EXIF)",
    )
    ap.add_argument(
        "--yaw-offset",
        type=float,
        default=0.0,
        help="yaw correction in degrees (clockwise). Use e.g. 90 or -90 if the mosaic is rotated.",
    )
    ap.add_argument(
        "--no-roi-warp",
        action="store_true",
        help="disable ROI warping (slower, uses full canvas warp per photo)",
    )
    ap.add_argument(
        "--roi-margin",
        type=int,
        default=8,
        help="ROI margin in pixels (default: 8)",
    )
    ap.add_argument(
        "--inspect-only",
        action="store_true",
        help="only inspect/print which EXIF/XMP parameters exist per file, then exit",
    )
    ap.add_argument(
        "--undistort",
        action="store_true",
        help="apply simple radial lens undistortion before mosaicking",
    )
    ap.add_argument(
        "--k1",
        type=float,
        default=0.1400,
        help="radial distortion k1 coefficient for --undistort (default: 0.1400)",
    )
    ap.add_argument(
        "--k2",
        type=float,
        default=-0.3979,
        help="radial distortion k2 coefficient for --undistort (default: -0.3979)",
    )
    ap.add_argument(
        "--k3",
        type=float,
        default=0.4837,
        help="radial distortion k3 coefficient for --undistort (default: 0.4837)",
    )
    ap.add_argument(
        "--yaw-invert",
        action="store_true",
        help="invert yaw sign (use -yaw). Useful if images rotate the wrong way.",
    )
    ap.add_argument(
        "--yaw-gimbal-only",
        action="store_true",
        help="use only Gimbal Yaw Degree for yaw (ignore Flight Yaw Degree)",
    )
    ap.add_argument(
        "--yaw-both",
        action="store_true",
        help="use Flight Yaw Degree + Gimbal Yaw Degree for yaw",
    )
    ap.add_argument(
        "--yaw-flight-only",
        action="store_true",
        help="use only Flight Yaw Degree for yaw (default behavior)",
    )
    ap.add_argument(
        "--gui",
        action="store_true",
        help="launch GUI for interactive adjustment (requires PyQt6)",
    )
    ap.add_argument(
        "--webui",
        action="store_true",
        help="launch experimental WebUI for GPU-accelerated preview in a browser",
    )
    ap.add_argument(
        "--webui-host",
        default="127.0.0.1",
        help="WebUI bind host (default: 127.0.0.1)",
    )
    ap.add_argument(
        "--webui-port",
        type=int,
        default=8765,
        help="WebUI bind port (default: 8765)",
    )
    ap.add_argument(
        "--webui-debug",
        action="store_true",
        help="print WebUI HTTP request logs to stderr",
    )
    ap.add_argument(
        "--preview-max-pix",
        type=int,
        default=3500,
        help="GUI/WebUI preview max width/height in pixels (default: 3500)",
    )
    ap.add_argument(
        "--use-pitch",
        action="store_true",
        help="use pitch_deg in geometry (enable tilt-based shift / cos correction)",
    )
    ap.add_argument(
        "--crop-optimize",
        action="store_true",
        help="save-only: in overlap, choose front pixel from image whose center is nearest",
    )
    return ap.parse_args(argv)


# ============================================================================
# Experimental WebUI Implementation
# ============================================================================

WEBUI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>bvpp WebUI</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #202124; background: #f6f7f8; }
    #app { display: grid; grid-template-rows: 1fr auto; height: 100%; }
    #stageWrap { position: relative; min-height: 0; background: #e8ebee; }
    #basemap, #gl, #overlay { position: absolute; inset: 0; display: block; width: 100%; height: 100%; }
    #basemap, #overlay { pointer-events: none; }
    #hudStack { position: absolute; left: 12px; top: 12px; display: grid; gap: 6px; pointer-events: none; }
    #hud { min-width: 250px; padding: 8px 10px; border-radius: 6px; background: rgba(255,255,255,.88); box-shadow: 0 1px 6px rgba(0,0,0,.16); font-size: 12px; line-height: 1.35; }
    #saveStatusPanel { display: none; min-width: 250px; max-width: 360px; padding: 7px 10px; border-radius: 6px; background: rgba(255,255,255,.9); box-shadow: 0 1px 6px rgba(0,0,0,.16); }
    #saveStatusPanel.active, #saveStatusPanel.has-status { display: block; }
    #rulerModeIndicator { position: absolute; right: 12px; top: 12px; display: none; padding: 7px 12px; border-radius: 999px; color: #174ea6; background: #e8f0fe; border: 1px solid #1a73e8; box-shadow: 0 1px 6px rgba(0,0,0,.16); font-size: 12px; font-weight: 700; letter-spacing: .04em; pointer-events: none; }
    #rulerModeIndicator.active { display: block; }
    #scaleIndicator { --scale-len: 120px; position: absolute; left: 12px; bottom: 10px; display: none; min-width: 72px; color: #202124; font-size: 12px; line-height: 1; pointer-events: none; }
    #scaleIndicator::before { content: ""; position: absolute; left: 0; bottom: 0; z-index: 0; width: calc(var(--scale-len) + 47px); height: calc(var(--scale-len) + 25px); border-radius: 5px; background: rgba(255,255,255,.86); box-shadow: 0 1px 5px rgba(0,0,0,.16); clip-path: polygon(0 0, 42px 0, 42px calc(100% - 19px), 100% calc(100% - 19px), 100% 100%, 0 100%); }
    .scale-indicator .scale-label, .scale-indicator .scale-bar, .scale-indicator .scale-v-label, .scale-indicator .scale-v-bar { z-index: 1; }
    .scale-indicator .scale-label { position: absolute; left: 40px; bottom: 0; height: 12px; font-weight: 600; white-space: nowrap; }
    .scale-indicator .scale-label span { position: absolute; top: 0; }
    .scale-indicator .scale-label .scale-label-mid { left: 50%; transform: translateX(-50%); }
    .scale-indicator .scale-label .scale-label-end { right: 0; }
    .scale-indicator .scale-bar { position: absolute; left: 40px; bottom: 9px; height: 10px; border: solid #202124; border-width: 2px 2px 0; box-sizing: border-box; }
    .scale-indicator .scale-v-label { position: absolute; left: 5px; bottom: 17px; width: 24px; font-weight: 600; white-space: nowrap; }
    .scale-indicator .scale-v-label span { position: absolute; left: 0; }
    .scale-indicator .scale-v-label .scale-label-mid { bottom: 50%; transform: translateY(50%); }
    .scale-indicator .scale-v-label .scale-label-end { top: 0; }
    .scale-indicator .scale-v-bar { position: absolute; left: 32px; bottom: 17px; width: 10px; border: solid #202124; border-width: 2px 2px 2px 0; box-sizing: border-box; }
    .scale-indicator .scale-tick { position: absolute; top: 0; width: 1px; height: 4px; background: rgba(32,33,36,.62); transform: translateX(-.5px); }
    .scale-indicator .scale-tick.major { height: 7px; background: rgba(32,33,36,.82); }
    .scale-indicator .scale-tick.strong { width: 2px; height: 10px; background: #202124; transform: translateX(-1px); }
    .scale-indicator .scale-v-bar .scale-tick { left: auto; right: 0; top: auto; width: 4px; height: 1px; transform: translateY(.5px); }
    .scale-indicator .scale-v-bar .scale-tick.major { width: 7px; height: 1px; }
    .scale-indicator .scale-v-bar .scale-tick.strong { width: 10px; height: 2px; transform: translateY(1px); }
    .hud-lines { white-space: pre-line; }
    .mem-row { margin-top: 6px; }
    .mem-bar { width: 100%; height: 8px; margin-top: 3px; overflow: hidden; border-radius: 999px; background: #dfe3e8; }
    .mem-fill { height: 100%; width: 0%; background: #188038; }
    #modalBackdrop { position: fixed; inset: 0; z-index: 20; display: none; align-items: center; justify-content: center; background: rgba(32,33,36,.38); }
    #modalBackdrop.active { display: flex; }
    #opacityDialog { width: min(520px, calc(100vw - 32px)); padding: 18px; border-radius: 8px; background: #fff; box-shadow: 0 12px 36px rgba(0,0,0,.28); }
    #opacityDialog h2 { margin: 0 0 8px; font-size: 16px; font-weight: 650; color: #202124; }
    #opacityDialog p { margin: 0 0 14px; color: #5f6368; font-size: 13px; line-height: 1.45; }
    .dialog-actions { display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }
    #controls { display: grid; gap: 8px; padding: 8px 12px 10px; border-top: 1px solid #d0d4d8; background: #fff; max-height: 38vh; overflow: auto; }
    .tabs { display: flex; gap: 4px; align-items: center; border-bottom: 1px solid #e1e5e9; }
    .tab-button { margin: 0 0 -1px; padding: 7px 10px; border: 0; border-bottom: 2px solid transparent; border-radius: 0; color: #5f6368; background: transparent; font-size: 12px; font-weight: 600; }
    .tab-button.active { color: #1a73e8; border-bottom-color: #1a73e8; background: #f8fbff; }
    .tab-panel { display: none; flex-wrap: wrap; align-items: stretch; gap: 10px; min-width: 0; }
    .tab-panel.active { display: flex; }
    fieldset { margin: 0; padding: 8px 10px 10px; border: 1px solid #d0d4d8; border-radius: 6px; min-width: 0; }
    legend { padding: 0 4px; color: #3c4043; font-size: 12px; font-weight: 600; }
    label { display: grid; gap: 4px; font-size: 12px; color: #5f6368; }
    .control-stack { display: grid; gap: 8px; align-content: start; }
    .adjustment-controls .control-stack { gap: 4px; }
    .adjustment-controls button { box-sizing: border-box; height: 22px; padding: 0 8px; font-size: 12px; line-height: 20px; }
    .ruler-controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .adjustment-controls .ruler-controls { gap: 4px; }
    .control-row { display: flex; align-items: end; gap: 8px; }
    .info-lines { white-space: pre-line; color: #5f6368; font-size: 12px; line-height: 1.35; }
    .compact { min-width: 180px; }
    .image-rotation { flex: 0 0 135px; min-width: 135px; }
    .image-opacity { flex: 0 0 150px; min-width: 150px; }
    .image-opacity input[type="range"] { width: 140px; }
    .image-overlap { flex: 0 0 130px; min-width: 130px; }
    .image-overlap .radio-stack { gap: 0; }
    .image-overlap .check { height: 24px; }
    .image-shortcuts { flex: 0 0 145px; min-width: 145px; }
    .save-options { flex: 0 0 220px; width: 220px; max-width: 220px; }
    .save-options .info-lines { white-space: normal; overflow-wrap: anywhere; }
    .wide { min-width: 290px; max-width: 360px; }
    .kmz-overlay { flex: 0 0 300px; max-width: 380px; }
    .kmz-overlay input[type="file"] { max-width: 100%; font-size: 12px; }
    .exif-group { flex: 1 1 180px; min-width: 170px; }
    .exif-product { flex-basis: 260px; }
    input, select, button { font: inherit; }
    input[type="text"] { width: 92px; padding: 5px 6px; }
    input[type="range"] { width: 160px; }
    select, button { padding: 6px 8px; }
    button { cursor: pointer; border: 1px solid #b9c0c7; border-radius: 6px; background: #fff; }
    button.active { border-color: #1a73e8; color: #174ea6; background: #e8f0fe; }
    button:disabled { cursor: default; opacity: .55; }
    .check { display: flex; align-items: center; gap: 6px; height: 30px; }
    .radio-stack { display: grid; gap: 5px; padding-left: 18px; }
    .attribution { color: #5f6368; font-size: 11px; line-height: 1.3; }
    #status { min-width: 220px; font-size: 12px; color: #5f6368; overflow-wrap: anywhere; }
    #altitudeShiftWarning { display: none; margin-top: 6px; max-width: 340px; font-size: 12px; line-height: 1.4; color: #d93025; font-weight: 650; overflow-wrap: anywhere; }
    #altitudeShiftWarning.active { display: block; }
    #saveActivity { display: none; width: 100%; height: 16px; margin-top: 4px; overflow: hidden; border-radius: 999px; background: #eef2f7; position: relative; }
    #saveActivity.active { display: block; }
    #saveActivity::before { content: ">>>"; position: absolute; left: -42px; top: 0; color: #1a73e8; font-size: 13px; line-height: 16px; font-weight: 700; letter-spacing: 3px; animation: save-arrow-run 1.05s linear infinite; }
    @keyframes save-arrow-run { from { transform: translateX(0); } to { transform: translateX(270px); } }
    #tooltip { position: fixed; z-index: 10; display: none; max-width: 420px; padding: 8px 10px; border-radius: 6px; background: rgba(32,33,36,.92); color: #fff; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.45; white-space: pre; pointer-events: none; box-shadow: 0 2px 10px rgba(0,0,0,.25); }
    #tooltip .altitude-alert { color: #ff6b6b; font-weight: 700; }
    #tooltip.message { white-space: normal; }
  </style>
</head>
<body>
  <div id="app">
    <div id="stageWrap">
      <canvas id="basemap"></canvas>
      <canvas id="gl" tabindex="0"></canvas>
      <canvas id="overlay"></canvas>
      <div id="hudStack">
        <div id="hud">Loading...</div>
        <div id="saveStatusPanel">
          <div id="status"></div>
          <div id="altitudeShiftWarning"></div>
          <div id="saveActivity"></div>
        </div>
      </div>
      <div id="rulerModeIndicator" role="status">EDIT RULER MODE</div>
      <div id="scaleIndicator" class="scale-indicator"><div class="scale-v-label"></div><div class="scale-v-bar"></div><div class="scale-label"></div><div class="scale-bar"></div></div>
      <div id="tooltip"></div>
    </div>
    <div id="controls">
      <div class="tabs" role="tablist" aria-label="Control panels">
        <button class="tab-button active" data-tab="image" type="button">Image Adjustment</button>
        <button class="tab-button" data-tab="misc" type="button">Misc. Control</button>
        <button class="tab-button" data-tab="save" type="button">Save</button>
        <button class="tab-button" data-tab="exif" type="button">Exif Info</button>
        <button class="tab-button" data-tab="area" type="button">Area</button>
      </div>
      <div class="tab-panel active" data-panel="image">
        <fieldset class="compact">
          <legend>Camera-Water Distance</legend>
          <div class="control-stack">
            <label>Altitude Correction [m]<input id="alt" type="text" inputmode="decimal"></label>
            <div id="cameraWaterInfo" class="info-lines"></div>
          </div>
        </fieldset>
        <fieldset class="compact image-rotation">
          <legend>Rotation Correction</legend>
          <label>Degree<input id="yaw" type="text" inputmode="decimal"></label>
        </fieldset>
        <fieldset class="compact image-opacity">
          <legend>Transparency (%)</legend>
          <label>Opacity <input id="opacity" type="range" min="0" max="100"><span id="opacityText"></span></label>
        </fieldset>
        <fieldset class="compact image-overlap">
          <legend>Overlap Effect</legend>
          <div class="radio-stack">
            <label class="check"><input type="radio" name="compareMode" value="normal" checked>None</label>
            <label class="check"><input type="radio" name="compareMode" value="pair-swipe">Swipe</label>
            <label class="check"><input type="radio" name="compareMode" value="pair-red-cyan">Red-Cyan</label>
          </div>
        </fieldset>
        <fieldset class="compact image-shortcuts">
          <legend>Keyboard Shortcuts</legend>
          <div id="shortcutInfo" class="info-lines"></div>
        </fieldset>
        <fieldset class="compact adjustment-controls">
          <legend>Adjustment Controls</legend>
          <div class="control-stack">
            <button id="fit">Fit to View</button>
            <div class="ruler-controls">
              <button id="rulerMode" type="button">Edit Ruler</button>
              <button id="clearRuler" type="button">Clear Ruler</button>
            </div>
            <button id="revert">Reset Adjustments</button>
          </div>
        </fieldset>
      </div>
      <div class="tab-panel" data-panel="save">
        <fieldset class="compact save-options">
          <legend>Save Options</legend>
          <label class="check"><input id="cropOptimize" type="checkbox">Crop Optimize</label>
          <div class="info-lines">When checked, mosaic saving uses the center area of each image as much as possible. This helps reduce blur near the edges of photos.</div>
        </fieldset>
        <fieldset class="compact">
          <legend>Actions</legend>
          <div class="control-stack">
            <div class="control-row">
              <button id="save">Save</button>
              <button id="cancelSave" disabled>Cancel</button>
            </div>
          </div>
        </fieldset>
      </div>
      <div class="tab-panel" data-panel="misc">
        <fieldset class="wide">
          <legend>Background Map</legend>
          <div class="control-stack">
            <label class="check"><input id="showBasemap" type="checkbox">Show Background Map</label>
            <div class="radio-stack">
              <label class="check"><input type="radio" name="basemapProvider" value="osm" checked>OpenStreetMap (Map)</label>
              <label class="check"><input type="radio" name="basemapProvider" value="gsi">日本国土地理院 (Photo)</label>
            </div>
          </div>
        </fieldset>
        <fieldset class="wide">
          <legend>Capture Order</legend>
          <label class="check"><input id="showCaptureOrder" type="checkbox">Show Capture Order</label>
          <div class="info-lines">Shows blue center markers, arrows from earlier to later captures, and capture time labels.</div>
        </fieldset>
        <fieldset class="wide">
          <legend>Image Rotation Rule</legend>
          <div class="control-stack">
            <select id="yawMode"><option value="both">Use Flight + Gimbal Yaw Degree</option><option value="flight_only">Use Flight Yaw Degree</option><option value="gimbal_only">Use Gimbal Yaw Degree</option></select>
            <label class="check"><input id="yawInvert" type="checkbox">Reverse rotate</label>
            <div id="yawDescription" class="info-lines"></div>
          </div>
        </fieldset>
        <fieldset class="wide kmz-overlay">
          <legend>KMZ Overlay</legend>
          <div class="control-stack">
            <input id="kmzFile" type="file" accept=".kmz,application/vnd.google-earth.kmz">
            <div class="control-row">
              <button id="clearKmz" type="button" disabled>Clear</button>
            </div>
            <div id="kmzInfo" class="info-lines"></div>
          </div>
        </fieldset>
      </div>
      <div class="tab-panel" data-panel="exif">
        <fieldset class="exif-group exif-product">
          <legend>Product/Lens</legend>
          <div id="exifProductInfo" class="info-lines"></div>
        </fieldset>
        <fieldset class="exif-group">
          <legend>Altitude</legend>
          <div id="relativeAltitudeInfo" class="info-lines"></div>
        </fieldset>
        <fieldset class="exif-group">
          <legend>Flight Speed</legend>
          <div id="exifSpeedInfo" class="info-lines"></div>
        </fieldset>
        <fieldset class="exif-group">
          <legend>Pitch</legend>
          <div id="exifPitchInfo" class="info-lines"></div>
        </fieldset>
        <fieldset class="exif-group">
          <legend>Roll</legend>
          <div id="exifRollInfo" class="info-lines"></div>
        </fieldset>
        <fieldset class="exif-group">
          <legend>Yaw</legend>
          <div id="exifYawInfo" class="info-lines"></div>
        </fieldset>
      </div>
      <div class="tab-panel" data-panel="area">
        <fieldset class="wide">
          <legend>Mosaic Area</legend>
          <div id="areaInfo" class="info-lines"></div>
        </fieldset>
      </div>
    </div>
  </div>
  <div id="modalBackdrop">
    <div id="opacityDialog" role="dialog" aria-modal="true" aria-labelledby="opacityDialogTitle">
      <h2 id="opacityDialogTitle">Save with transparency?</h2>
      <p id="opacityDialogMessage"></p>
      <div class="dialog-actions">
        <button id="opacityDialogSave" class="primary">Save</button>
        <button id="opacityDialogSaveOpaque">Save without transparency</button>
        <button id="opacityDialogCancel">Cancel</button>
      </div>
    </div>
  </div>
  <script src="/app.js"></script>
</body>
</html>
"""


WEBUI_JS = r"""(() => {
  const preferredLang = (Array.isArray(navigator.languages) && navigator.languages.length ? navigator.languages[0] : navigator.language) || 'en';
  const uiLang = String(preferredLang).toLowerCase().startsWith('ja') ? 'ja' : 'en';
  const I18N = {
    en: {
      appTitle: 'bvpp WebUI', loading: 'Loading...', controlPanels: 'Control panels', imageAdjustment: 'Image Adjustment', miscControl: 'Misc. Control', save: 'Save', cancel: 'Cancel', exifInfo: 'Exif Info', area: 'Area',
      cameraWaterDistance: 'Camera-Water Distance', altitudeCorrectionM: 'Altitude Correction [m]', rotationCorrection: 'Rotation Correction', degree: 'Degree', transparencyPct: 'Transparency (%)', opacity: 'Opacity',
      overlapEffect: 'Overlap Effect', none: 'None', swipe: 'Swipe', redCyan: 'Red-Cyan', keyboardShortcuts: 'Keyboard Shortcuts', adjustmentControls: 'Adjustment Controls',
      fitToView: 'Fit to View', editRuler: 'Edit Ruler', clearRuler: 'Clear Ruler', resetAdjustments: 'Reset Adjustments', saveOptions: 'Save Options', cropOptimize: 'Crop Optimize',
      cropOptimizeHelp: 'When checked, mosaic saving uses the center area of each image as much as possible. This helps reduce blur near the edges of photos.', actions: 'Actions', backgroundMap: 'Background Map',
      showBackgroundMap: 'Show Background Map', osmMap: 'OpenStreetMap (Map)', gsiPhoto: '日本国土地理院 (Photo)', captureOrder: 'Capture Order', showCaptureOrder: 'Show Capture Order',
      captureOrderHelp: 'Shows blue center markers, arrows from earlier to later captures, and capture time labels.', imageRotationRule: 'Image Rotation Rule', yawBoth: 'Use Flight + Gimbal Yaw Degree',
      yawFlightOnly: 'Use Flight Yaw Degree', yawGimbalOnly: 'Use Gimbal Yaw Degree', reverseRotate: 'Reverse rotate', productLens: 'Product/Lens', altitude: 'Altitude', flightSpeed: 'Flight Speed', pitch: 'Pitch', roll: 'Roll', yaw: 'Yaw', mosaicArea: 'Mosaic Area',
      kmzOverlay: 'KMZ Overlay', clear: 'Clear', loadingKmz: 'Loading KMZ...', kmzLoaded: 'Loaded: {name} ({count} feature(s))', kmzCleared: 'KMZ overlay cleared', kmzNoOverlay: 'No displayable Point or GroundOverlay found in this KMZ.', kmzLoadFailed: 'KMZ load failed: {error}',
      saveWithTransparencyTitle: 'Save with transparency?', saveWithoutTransparency: 'Save without transparency', insufficientMemoryTitle: 'Insufficient memory may be', forceSave: 'Force Save', saveFailedTitle: 'Save failed', ok: 'OK', editRulerMode: 'EDIT RULER MODE',
      altitudeShiftPanelWarning: 'The offset between Relative Altitude and the actual altitude appears to have changed during the flight. In this case, the required altitude correction is likely to differ by location. Check "Show Capture Order" to identify where the offset changed, split the images before and after each change into separate folders, and process them separately.',
      altitudeShiftArrowWarning: 'The actual altitude appears to have diverged from Relative Altitude. We recommend moving this and subsequent images to a separate folder and processing them separately.', approxPrefix: 'Approx.', mosaicImageArea: 'Mosaic Image Area', overlappingArea: 'Overlapping Area', overlapRatio: 'Overlap Ratio',
      waitingForAdjustment: 'Waiting for adjustment to stop...', calculating: 'Calculating...', areaCalculationFailed: 'Area calculation failed: {error}', rulerStart: 'Ruler: click the start point on an image.', rulerCleared: 'Ruler cleared', rulerEnd: 'Ruler: click the end point.', rulerValue: 'Ruler: {value}',
      file: 'File', exif: 'Exif', relativeAltitudeM: 'Relative Altitude [m]', absoluteAltitudeM: 'Absolute Altitude [m]', absRelAltM: 'Abs - Rel Alt. [m]', flightPitchDegree: 'Flight Pitch [degree]', gimbalPitchDegree: 'Gimbal Pitch [degree]', flightRollDegreeLabel: 'Flight Roll [degree]', flightYawDegree: 'Flight Yaw Degree', gimbalYawDegree: 'Gimbal Yaw Degree', flightXYZSpeed: 'Flight X/Y/Z Speed', webglUnavailable: 'WebGL is not available in this browser.', loadedTextures: 'Loaded {loaded}/{total} textures', readyTextures: 'Ready: {loaded}/{total} textures', opacitySaveMessage: 'Opacity is currently {opacity}%. Choose how to save the mosaic.', memoryForceMessage: 'There may be about {shortage} less available memory than required for saving. A shortage of around 5 GB may still be possible to run, but anything beyond that depends on the situation. Do you want to continue?',
      unknownError: 'unknown error', loadingTextures: 'Loading {count} textures...', photos: 'photos', preview: 'preview', saveSize: 'save', estimatedMemoryRequired: 'estimated memory required for saving', zoom: 'zoom', physicalMemory: 'physical memory', availableMemory: 'available memory',
      updatingGeometry: 'Updating geometry...', readyGeometryUpdated: 'Ready: geometry updated', updateFailed: 'Update failed: {error}', reverting: 'Reverting...', reverted: 'Reverted to original', saving: 'Saving...', saveFailed: 'Save failed: {error}', saved: 'Saved: {path}', saveCancelled: 'Save cancelled', saveStatusFailed: 'Save status failed: {error}', startingSave: 'Starting full-resolution mosaic save...', saveStartFailed: 'Save start failed: {error}', cancellingSave: 'Cancelling save...',
      preparing: 'Preparing', computingCanvas: 'Computing canvas', processing: 'Processing', savingPhase: 'Saving', done: 'Done', cancelled: 'Cancelled', error: 'Error', saveProgressElapsed: 'Saving... elapsed {elapsed}s, size {size}, speed {speed} MB/s',
      avgCameraWaterDistance: 'Avg. Camera-Water Distance', relativeAvg: 'Relative Avg', relativeMin: 'Relative Min', relativeMax: 'Relative Max', absoluteAvg: 'Absolute Avg', absoluteMin: 'Absolute Min', absoluteMax: 'Absolute Max',
      flightYawDegreeDesc: 'Flight Yaw Degree = Drone direction (Exif)', gimbalYawDegreeDesc: 'Gimbal Yaw Degree = Gimbal direction (Exif)', shortcutAltitudeUp: 'H: Altitude +0.2m', shortcutAltitudeDown: 'J: Altitude -0.2m', shortcutRotationUp: 'K: Rotation +2 deg', shortcutRotationDown: 'L: Rotation -2 deg', noExifStats: 'No EXIF stats available',
      productName: 'Product Name', uniqueCameraModel: 'Unique Camera Model', uniform: 'uniform', mismatchDetected: 'mismatch detected', flightPitchDeg: 'Flight Pitch [deg]', gimbalPitchDeg: 'Gimbal Pitch [deg]', gimbalRollDegree: 'Gimbal Roll [degree]', flightRollDegree: 'Flight Roll [degree]', flightYawDeg: 'Flight Yaw [deg]', gimbalYawDeg: 'Gimbal Yaw [deg]', flightXSpeedMps: 'Flight X Speed [m/s]', flightYSpeedMps: 'Flight Y Speed [m/s]', flightZSpeedMps: 'Flight Z Speed [m/s]', sourceGsi: 'Source: 国土地理院 地理院タイル'
    },
    ja: {
      appTitle: 'bvpp WebUI', loading: '読み込み中...', controlPanels: '操作パネル', imageAdjustment: '画像調整', miscControl: 'その他の操作', save: '保存', cancel: 'キャンセル', exifInfo: 'Exif情報', area: '面積',
      cameraWaterDistance: 'カメラ-水面距離', altitudeCorrectionM: '高度補正 [m]', rotationCorrection: '回転補正', degree: '角度', transparencyPct: '透過率 (%)', opacity: '不透明度',
      overlapEffect: '重なり表示効果', none: 'なし', swipe: 'スワイプ', redCyan: '赤-シアン', keyboardShortcuts: 'キーボードショートカット', adjustmentControls: '調整コントロール',
      fitToView: '全体表示', editRuler: 'ルーラー編集', clearRuler: 'ルーラー消去', resetAdjustments: '調整をリセット', saveOptions: '保存オプション', cropOptimize: 'クロップ最適化',
      cropOptimizeHelp: 'オンにすると、モザイク保存時に各画像の中央付近をできるだけ使います。写真の端付近のぼけを減らすのに役立ちます。', actions: '操作', backgroundMap: '背景地図',
      showBackgroundMap: '背景地図を表示', osmMap: 'OpenStreetMap (地図)', gsiPhoto: '日本国土地理院 (写真)', captureOrder: '撮影順', showCaptureOrder: '撮影順を表示',
      captureOrderHelp: '青い中心マーカー、撮影の前後を示す矢印、撮影時刻ラベルを表示します。', imageRotationRule: '画像回転ルール', yawBoth: '飛行Yaw角 + ジンバルYaw角を使用',
      yawFlightOnly: '飛行Yaw角を使用', yawGimbalOnly: 'ジンバルYaw角を使用', reverseRotate: '逆回転', productLens: '製品/レンズ', altitude: '高度', flightSpeed: '飛行速度', pitch: 'ピッチ', roll: 'ロール', yaw: 'Yaw', mosaicArea: 'モザイク面積',
      kmzOverlay: 'KMZ読み込み', clear: 'クリア', loadingKmz: 'KMZを読み込み中...', kmzLoaded: '読み込み完了: {name} ({count} 件)', kmzCleared: 'KMZ表示を消去しました', kmzNoOverlay: 'このKMZには表示可能なPointまたはGroundOverlayがありません。', kmzLoadFailed: 'KMZ読み込みに失敗しました: {error}',
      saveWithTransparencyTitle: '透過ありで保存しますか？', saveWithoutTransparency: '透過なしで保存', insufficientMemoryTitle: 'メモリが不足している可能性があります', forceSave: '強制保存', saveFailedTitle: '保存に失敗しました', ok: 'OK', editRulerMode: 'ルーラー編集モード',
      altitudeShiftPanelWarning: 'Relative Altitude と実高度の差が飛行中に変化しているようです。この場合、必要な高度補正は場所によって異なる可能性があります。「撮影順を表示」でずれが変わった位置を確認し、変化の前後で画像を別フォルダに分けて、それぞれ処理してください。',
      altitudeShiftArrowWarning: '実高度と Relative Altitude の差が変化したようです。この画像以降を別フォルダに移して、分けて処理することを推奨します。', approxPrefix: '概算', mosaicImageArea: 'モザイク画像面積', overlappingArea: '重複面積', overlapRatio: '重複率',
      waitingForAdjustment: '調整が止まるのを待っています...', calculating: '計算中...', areaCalculationFailed: '面積計算に失敗しました: {error}', rulerStart: 'ルーラー: 画像上の開始点をクリックしてください。', rulerCleared: 'ルーラーを消去しました', rulerEnd: 'ルーラー: 終了点をクリックしてください。', rulerValue: 'ルーラー: {value}',
      file: 'ファイル', exif: 'Exif', relativeAltitudeM: 'Relative Altitude [m]', absoluteAltitudeM: 'Absolute Altitude [m]', absRelAltM: 'Abs - Rel Alt. [m]', flightPitchDegree: '飛行ピッチ [degree]', gimbalPitchDegree: 'ジンバルピッチ [degree]', flightRollDegreeLabel: '飛行ロール [degree]', flightYawDegree: '飛行Yaw角', gimbalYawDegree: 'ジンバルYaw角', flightXYZSpeed: '飛行X/Y/Z速度', webglUnavailable: 'このブラウザでは WebGL を利用できません。', loadedTextures: '{loaded}/{total} 個のテクスチャを読み込みました', readyTextures: '準備完了: {loaded}/{total} 個のテクスチャ', opacitySaveMessage: '現在の不透明度は {opacity}% です。モザイクの保存方法を選択してください。', memoryForceMessage: '保存に必要な量に対して、利用可能メモリが約 {shortage} 不足している可能性があります。5 GB 程度の不足なら実行できる場合もありますが、状況に依存します。続行しますか？',
      unknownError: '不明なエラー', loadingTextures: '{count} 個のテクスチャを読み込み中...', photos: '枚の写真', preview: 'プレビュー', saveSize: '保存', estimatedMemoryRequired: '保存に必要な推定メモリ', zoom: 'ズーム', physicalMemory: '物理メモリ', availableMemory: '利用可能メモリ',
      updatingGeometry: 'ジオメトリを更新中...', readyGeometryUpdated: '準備完了: ジオメトリを更新しました', updateFailed: '更新に失敗しました: {error}', reverting: '戻しています...', reverted: '元の状態に戻しました', saving: '保存中...', saveFailed: '保存に失敗しました: {error}', saved: '保存しました: {path}', saveCancelled: '保存をキャンセルしました', saveStatusFailed: '保存状態の取得に失敗しました: {error}', startingSave: 'フル解像度モザイク保存を開始しています...', saveStartFailed: '保存開始に失敗しました: {error}', cancellingSave: '保存をキャンセル中...',
      preparing: '準備中', computingCanvas: 'キャンバスを計算中', processing: '処理中', savingPhase: '保存中', done: '完了', cancelled: 'キャンセル済み', error: 'エラー', saveProgressElapsed: '保存中... 経過 {elapsed}s, サイズ {size}, 速度 {speed} MB/s',
      avgCameraWaterDistance: '平均カメラ-水面距離', relativeAvg: 'Relative 平均', relativeMin: 'Relative 最小', relativeMax: 'Relative 最大', absoluteAvg: 'Absolute 平均', absoluteMin: 'Absolute 最小', absoluteMax: 'Absolute 最大',
      flightYawDegreeDesc: 'Flight Yaw Degree = ドローンの向き (Exif)', gimbalYawDegreeDesc: 'Gimbal Yaw Degree = ジンバルの向き (Exif)', shortcutAltitudeUp: 'H: 高度 +0.2m', shortcutAltitudeDown: 'J: 高度 -0.2m', shortcutRotationUp: 'K: 回転 +2 deg', shortcutRotationDown: 'L: 回転 -2 deg', noExifStats: 'EXIF統計情報はありません',
      productName: '製品名', uniqueCameraModel: '固有カメラモデル', uniform: '均一', mismatchDetected: '不一致を検出', flightPitchDeg: '飛行ピッチ [deg]', gimbalPitchDeg: 'ジンバルピッチ [deg]', gimbalRollDegree: 'ジンバルロール [degree]', flightRollDegree: '飛行ロール [degree]', flightYawDeg: '飛行Yaw [deg]', gimbalYawDeg: 'ジンバルYaw [deg]', flightXSpeedMps: '飛行X速度 [m/s]', flightYSpeedMps: '飛行Y速度 [m/s]', flightZSpeedMps: '飛行Z速度 [m/s]', sourceGsi: '出典: 国土地理院 地理院タイル'
    }
  };
  function t(key, vars = {}) {
    let text = (I18N[uiLang] && I18N[uiLang][key]) || key;
    for (const [name, value] of Object.entries(vars)) text = text.replaceAll(`{${name}}`, String(value));
    return text;
  }
  function translatePhase(phase) {
    const phases = { Preparing: 'preparing', 'Computing canvas': 'computingCanvas', Processing: 'processing', Saving: 'savingPhase', Done: 'done', Cancelled: 'cancelled', Error: 'error' };
    return t(phases[phase] || phase);
  }
  function translateSaveText(text) {
    const m = String(text || '').match(/^Saving\.\.\. elapsed ([0-9.]+)s, size (.*), speed ([0-9.]+) MB\/s$/);
    if (!m) return text || t('saving');
    return t('saveProgressElapsed', { elapsed: m[1], size: m[2], speed: m[3] });
  }
  function translateInfoLine(line) {
    if (uiLang !== 'ja') return line;
    let text = String(line || '');
    const exact = { 'Flight Yaw Degree = Drone direction (Exif)': t('flightYawDegreeDesc'), 'Gimbal Yaw Degree = Gimbal direction (Exif)': t('gimbalYawDegreeDesc'), 'H: Altitude +0.2m': t('shortcutAltitudeUp'), 'J: Altitude -0.2m': t('shortcutAltitudeDown'), 'K: Rotation +2 deg': t('shortcutRotationUp'), 'L: Rotation -2 deg': t('shortcutRotationDown'), 'No EXIF stats available': t('noExifStats') };
    if (exact[text]) return exact[text];
    const prefixMap = [['Avg. Camera-Water Distance', t('avgCameraWaterDistance')], ['Relative Avg', t('relativeAvg')], ['Relative Min', t('relativeMin')], ['Relative Max', t('relativeMax')], ['Absolute Avg', t('absoluteAvg')], ['Absolute Min', t('absoluteMin')], ['Absolute Max', t('absoluteMax')], ['Product Name', t('productName')], ['Unique Camera Model', t('uniqueCameraModel')], ['Flight Pitch [deg]', t('flightPitchDeg')], ['Gimbal Pitch [deg]', t('gimbalPitchDeg')], ['Gimbal Roll [degree]', t('gimbalRollDegree')], ['Flight Roll [degree]', t('flightRollDegree')], ['Flight Yaw [deg]', t('flightYawDeg')], ['Gimbal Yaw [deg]', t('gimbalYawDeg')], ['Flight X Speed [m/s]', t('flightXSpeedMps')], ['Flight Y Speed [m/s]', t('flightYSpeedMps')], ['Flight Z Speed [m/s]', t('flightZSpeedMps')]];
    for (const [from, to] of prefixMap) if (text.startsWith(`${from}:`)) return text.replace(from, to);
    if (text.startsWith('FOV [deg]: mismatch detected')) return text.replace('mismatch detected', t('mismatchDetected'));
    return text.replace('(uniform)', `(${t('uniform')})`);
  }
  function translateInfoLines(lines) { return (Array.isArray(lines) ? lines : []).map(translateInfoLine); }
  function setText(selector, key) {
    const el = document.querySelector(selector);
    if (!el) return;
    let replaced = false;
    el.childNodes.forEach(n => {
      if (n.nodeType !== Node.TEXT_NODE) return;
      if (!replaced) {
        n.textContent = t(key);
        replaced = true;
      } else {
        n.textContent = '';
      }
    });
    if (!replaced) el.appendChild(document.createTextNode(t(key)));
  }
  function applyStaticI18n() {
    document.documentElement.lang = uiLang;
    document.title = t('appTitle');
    const tabLabels = { image: 'imageAdjustment', misc: 'miscControl', save: 'save', exif: 'exifInfo', area: 'area' };
    document.querySelector('.tabs')?.setAttribute('aria-label', t('controlPanels'));
    document.querySelectorAll('.tab-button').forEach(btn => { const key = tabLabels[btn.dataset.tab]; if (key) btn.textContent = t(key); });
    setText('#rulerModeIndicator', 'editRulerMode');
    [['.tab-panel[data-panel="image"] fieldset:nth-of-type(1) legend','cameraWaterDistance'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(1) label','altitudeCorrectionM'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(2) legend','rotationCorrection'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(2) label','degree'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(3) legend','transparencyPct'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(3) label','opacity'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(4) legend','overlapEffect'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(4) label:nth-child(1)','none'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(4) label:nth-child(2)','swipe'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(4) label:nth-child(3)','redCyan'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(5) legend','keyboardShortcuts'],['.tab-panel[data-panel="image"] fieldset:nth-of-type(6) legend','adjustmentControls'],['#fit','fitToView'],['#rulerMode','editRuler'],['#clearRuler','clearRuler'],['#revert','resetAdjustments'],['.tab-panel[data-panel="save"] fieldset:nth-of-type(1) legend','saveOptions'],['.tab-panel[data-panel="save"] fieldset:nth-of-type(1) label','cropOptimize'],['.tab-panel[data-panel="save"] fieldset:nth-of-type(1) .info-lines','cropOptimizeHelp'],['.tab-panel[data-panel="save"] fieldset:nth-of-type(2) legend','actions'],['#save','save'],['#cancelSave','cancel'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(1) legend','backgroundMap'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(1) .control-stack > label','showBackgroundMap'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(1) .radio-stack label:nth-child(1)','osmMap'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(1) .radio-stack label:nth-child(2)','gsiPhoto'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(2) legend','captureOrder'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(2) label','showCaptureOrder'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(2) .info-lines','captureOrderHelp'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(3) legend','imageRotationRule'],['#yawMode option[value="both"]','yawBoth'],['#yawMode option[value="flight_only"]','yawFlightOnly'],['#yawMode option[value="gimbal_only"]','yawGimbalOnly'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(3) label','reverseRotate'],['.tab-panel[data-panel="misc"] fieldset:nth-of-type(4) legend','kmzOverlay'],['#clearKmz','clear'],['.tab-panel[data-panel="exif"] fieldset:nth-of-type(1) legend','productLens'],['.tab-panel[data-panel="exif"] fieldset:nth-of-type(2) legend','altitude'],['.tab-panel[data-panel="exif"] fieldset:nth-of-type(3) legend','flightSpeed'],['.tab-panel[data-panel="exif"] fieldset:nth-of-type(4) legend','pitch'],['.tab-panel[data-panel="exif"] fieldset:nth-of-type(5) legend','roll'],['.tab-panel[data-panel="exif"] fieldset:nth-of-type(6) legend','yaw'],['.tab-panel[data-panel="area"] legend','mosaicArea'],['#opacityDialogTitle','saveWithTransparencyTitle'],['#opacityDialogSave','save'],['#opacityDialogSaveOpaque','saveWithoutTransparency'],['#opacityDialogCancel','cancel']].forEach(([selector, key]) => setText(selector, key));
    if (hud.textContent === 'Loading...') hud.textContent = t('loading');
  }
  const basemapCanvas = document.getElementById('basemap');
  const canvas = document.getElementById('gl');
  const overlayCanvas = document.getElementById('overlay');
  const hud = document.getElementById('hud');
  const scaleIndicator = document.getElementById('scaleIndicator');
  const saveStatusPanel = document.getElementById('saveStatusPanel');
  const tooltip = document.getElementById('tooltip');
  const statusEl = document.getElementById('status');
  const altitudeShiftWarningEl = document.getElementById('altitudeShiftWarning');
  const saveActivityEl = document.getElementById('saveActivity');
  const altEl = document.getElementById('alt');
  const yawEl = document.getElementById('yaw');
  const yawModeEl = document.getElementById('yawMode');
  const yawInvertEl = document.getElementById('yawInvert');
  const opacityEl = document.getElementById('opacity');
  const opacityTextEl = document.getElementById('opacityText');
  const compareModeEls = Array.from(document.querySelectorAll('input[name="compareMode"]'));
  const rulerModeBtn = document.getElementById('rulerMode');
  const clearRulerBtn = document.getElementById('clearRuler');
  const rulerModeIndicatorEl = document.getElementById('rulerModeIndicator');
  const cropOptimizeEl = document.getElementById('cropOptimize');
  const showCaptureOrderEl = document.getElementById('showCaptureOrder');
  const showBasemapEl = document.getElementById('showBasemap');
  const basemapProviderEls = Array.from(document.querySelectorAll('input[name="basemapProvider"]'));
  const kmzFileEl = document.getElementById('kmzFile');
  const clearKmzBtn = document.getElementById('clearKmz');
  const kmzInfoEl = document.getElementById('kmzInfo');
  const saveBtn = document.getElementById('save');
  const cancelSaveBtn = document.getElementById('cancelSave');
  const modalBackdrop = document.getElementById('modalBackdrop');
  const opacityDialogTitle = document.getElementById('opacityDialogTitle');
  const opacityDialogMessage = document.getElementById('opacityDialogMessage');
  const opacityDialogSave = document.getElementById('opacityDialogSave');
  const opacityDialogSaveOpaque = document.getElementById('opacityDialogSaveOpaque');
  const opacityDialogCancel = document.getElementById('opacityDialogCancel');
  const cameraWaterInfoEl = document.getElementById('cameraWaterInfo');
  const relativeAltitudeInfoEl = document.getElementById('relativeAltitudeInfo');
  const yawDescriptionEl = document.getElementById('yawDescription');
  const shortcutInfoEl = document.getElementById('shortcutInfo');
  const exifProductInfoEl = document.getElementById('exifProductInfo');
  const exifSpeedInfoEl = document.getElementById('exifSpeedInfo');
  const exifPitchInfoEl = document.getElementById('exifPitchInfo');
  const exifRollInfoEl = document.getElementById('exifRollInfo');
  const exifYawInfoEl = document.getElementById('exifYawInfo');
  const areaInfoEl = document.getElementById('areaInfo');
  const tabButtons = Array.from(document.querySelectorAll('.tab-button'));
  const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
  const basemapCtx = basemapCanvas.getContext('2d');
  const overlayCtx = overlayCanvas.getContext('2d');
  const gl = canvas.getContext('webgl', { antialias: true, alpha: true });
  if (!gl) {
    hud.textContent = t('webglUnavailable');
    return;
  }

  const vs = `
    attribute vec2 a_pos;
    attribute vec2 a_uv;
    uniform vec2 u_canvas;
    uniform vec2 u_view;
    uniform float u_zoom;
    uniform vec2 u_pan;
    varying vec2 v_uv;
    void main() {
      vec2 p = (a_pos - u_pan) * u_zoom;
      vec2 clip = vec2((p.x / u_view.x) * 2.0 - 1.0, 1.0 - (p.y / u_view.y) * 2.0);
      gl_Position = vec4(clip, 0.0, 1.0);
      v_uv = a_uv;
    }`;
  const fs = `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_tex;
    uniform float u_opacity;
    uniform int u_compare_mode;
    uniform float u_layer_parity;
    uniform float u_swipe_x;
    void main() {
      if (u_compare_mode == 1) {
        bool showLeft = gl_FragCoord.x <= u_swipe_x;
        if ((u_layer_parity < 0.5 && !showLeft) || (u_layer_parity > 0.5 && showLeft)) discard;
      }
      vec4 c = texture2D(u_tex, v_uv);
      if (u_compare_mode == 2) {
        float gray = dot(c.rgb, vec3(0.299, 0.587, 0.114));
        vec3 falseColor = u_layer_parity < 0.5 ? vec3(gray, 0.0, 0.0) : vec3(0.0, gray, gray);
        gl_FragColor = vec4(falseColor, c.a * u_opacity);
        return;
      }
      gl_FragColor = vec4(c.rgb, c.a * u_opacity);
    }`;

  function shader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);

  const loc = {
    pos: gl.getAttribLocation(program, 'a_pos'),
    uv: gl.getAttribLocation(program, 'a_uv'),
    canvas: gl.getUniformLocation(program, 'u_canvas'),
    view: gl.getUniformLocation(program, 'u_view'),
    zoom: gl.getUniformLocation(program, 'u_zoom'),
    pan: gl.getUniformLocation(program, 'u_pan'),
    opacity: gl.getUniformLocation(program, 'u_opacity'),
    compareMode: gl.getUniformLocation(program, 'u_compare_mode'),
    layerParity: gl.getUniformLocation(program, 'u_layer_parity'),
    swipeX: gl.getUniformLocation(program, 'u_swipe_x'),
  };
  const posBuf = gl.createBuffer();
  const uvBuf = gl.createBuffer();
  const uvData = new Float32Array([0,0, 1,0, 1,1, 0,0, 1,1, 0,1]);
  gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
  gl.bufferData(gl.ARRAY_BUFFER, uvData, gl.STATIC_DRAW);

  let state = null;
  let layers = [];
  let zoom = 1;
  let pan = [0, 0];
  let dragging = false;
  let lastMouse = [0, 0];
  let loadGeneration = 0;
  let savePollTimer = null;
  let memoryPollTimer = null;
  let memoryPollInterval = 0;
  let memoryInfo = null;
  let hoverPhotoId = null;
  let hoverUnderPhotoId = null;
  let hoverScreenX = 0;
  let shutdownSent = false;
  let lastCommittedNumberText = "";
  let mouseDownPos = null;
  let mouseDragged = false;
  let rulerMode = false;
  let ruler = null;
  let rulerDrag = null;
  let rulerSuppressClick = false;
  let shownSaveError = null;
  let areaStatsTimer = null;
  let areaStatsGeneration = 0;
  let kmzOverlay = null;
  let kmzLoadGeneration = 0;
  const areaStatsIdleMs = 500;
  const tileCache = new Map();
  const altitudeShiftPanelWarningText = t('altitudeShiftPanelWarning');
  const altitudeShiftArrowWarningText = t('altitudeShiftArrowWarning');

  function refreshSaveStatusPanel() {
    if (!saveStatusPanel || !statusEl || !altitudeShiftWarningEl || !saveActivityEl) return;
    saveStatusPanel.classList.toggle('has-status', !!statusEl.textContent || altitudeShiftWarningEl.classList.contains('active'));
    saveStatusPanel.classList.toggle('active', saveActivityEl.classList.contains('active'));
  }
  function setStatus(text) {
    statusEl.textContent = text || '';
    refreshSaveStatusPanel();
  }
  function updateAltitudeShiftWarning() {
    if (!altitudeShiftWarningEl) return;
    const hasWarning = !!(state && Array.isArray(state.sequence) && state.sequence.some(item => item.altitude_shift_from_previous));
    altitudeShiftWarningEl.textContent = hasWarning ? altitudeShiftPanelWarningText : '';
    altitudeShiftWarningEl.classList.toggle('active', hasWarning);
    refreshSaveStatusPanel();
  }
  function basemapProvider() {
    const selected = basemapProviderEls.find(el => el.checked);
    return selected ? selected.value : 'osm';
  }
  function basemapEnabled() {
    return !!(showBasemapEl && showBasemapEl.checked);
  }
  function captureOrderEnabled() {
    return !!(showCaptureOrderEl && showCaptureOrderEl.checked);
  }
  function notifyServerShutdown() {
    if (shutdownSent) return;
    shutdownSent = true;
    try {
      navigator.sendBeacon('/api/shutdown', new Blob(['{}'], { type: 'application/json' }));
    } catch (err) {
      try {
        fetch('/api/shutdown', { method: 'POST', keepalive: true, headers: {'Content-Type': 'application/json'}, body: '{}' });
      } catch (_) {}
    }
  }
  function setSaveActive(active, cancellable = active) {
    saveBtn.disabled = !!active;
    cancelSaveBtn.disabled = !cancellable;
    saveActivityEl.classList.toggle('active', !!active);
    refreshSaveStatusPanel();
    setMemoryPolling(active ? 1000 : 5000);
  }
  function setLines(el, lines) {
    if (!el) return;
    el.textContent = Array.isArray(lines) ? lines.join('\n') : String(lines || '');
  }
  function updateExifGroups(lines) {
    const product = [];
    const speed = [];
    const pitch = [];
    const roll = [];
    const yaw = [];
    for (const line of Array.isArray(lines) ? lines : []) {
      if (/^(Product Name|Unique Camera Model|FOV)\b/.test(line)) product.push(line);
      else if (/^Flight [XYZ] Speed\b/.test(line)) speed.push(line);
      else if (/Pitch\b/.test(line)) pitch.push(line);
      else if (/Roll\b/.test(line)) roll.push(line);
      else if (/Yaw\b/.test(line)) yaw.push(line);
      else product.push(line);
    }
    setLines(exifProductInfoEl, translateInfoLines(product));
    setLines(exifSpeedInfoEl, translateInfoLines(speed));
    setLines(exifPitchInfoEl, translateInfoLines(pitch));
    setLines(exifRollInfoEl, translateInfoLines(roll));
    setLines(exifYawInfoEl, translateInfoLines(yaw));
  }
  function updateInfo() {
    if (!state || !state.info) return;
    setLines(cameraWaterInfoEl, translateInfoLines(state.info.camera_water_lines));
    setLines(relativeAltitudeInfoEl, translateInfoLines(state.info.relative_altitude_lines));
    setLines(yawDescriptionEl, translateInfoLines(state.info.yaw_description_lines));
    setLines(shortcutInfoEl, translateInfoLines(state.info.shortcut_lines));
    updateExifGroups(state.info.exif_lines);
    updateAltitudeShiftWarning();
  }
  function showAreaStats(area) {
    const areaPrefix = area.approximate ? `${t('approxPrefix')} ` : '';
    setLines(areaInfoEl, [
      `${areaPrefix}${t('mosaicImageArea')}: ${formatAreaWithError(area.mosaic_area_m2, area.area_error_m2)}`,
      `${areaPrefix}${t('overlappingArea')}: ${formatAreaWithError(area.overlap_area_m2, area.area_error_m2)}`,
      `${areaPrefix}${t('overlapRatio')}: ${formatPercentWithError(area.overlap_pct, area.overlap_pct_error)}`,
    ]);
  }
  function isAreaTabActive() {
    return tabPanels.some(panel => panel.dataset.panel === 'area' && panel.classList.contains('active'));
  }
  function invalidateAreaStats() {
    areaStatsGeneration += 1;
    clearTimeout(areaStatsTimer);
    areaStatsTimer = null;
    if (isAreaTabActive()) setLines(areaInfoEl, t('waitingForAdjustment'));
  }
  async function loadAreaStats(generation) {
    if (!isAreaTabActive() || generation !== areaStatsGeneration) return;
    setLines(areaInfoEl, t('calculating'));
    try {
      const res = await fetch('/api/area-stats');
      const area = await res.json();
      if (!isAreaTabActive() || generation !== areaStatsGeneration) return;
      showAreaStats(area);
    } catch (err) {
      if (isAreaTabActive() && generation === areaStatsGeneration) {
        setLines(areaInfoEl, t('areaCalculationFailed', { error: err }));
      }
    }
  }
  function scheduleAreaStats() {
    if (!isAreaTabActive()) return;
    clearTimeout(areaStatsTimer);
    const generation = ++areaStatsGeneration;
    setLines(areaInfoEl, t('waitingForAdjustment'));
    areaStatsTimer = setTimeout(() => loadAreaStats(generation), areaStatsIdleMs);
  }
  function photoById(id) {
    if (!state || !state.photos) return null;
    return state.photos.find(photo => photo.id === id) || null;
  }
  function fmtNum(value) {
    return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(2) : 'N/A';
  }
  function formatArea(value) {
    const area = Number(value);
    if (!Number.isFinite(area)) return 'N/A';
    if (area >= 1000000) return `${(area / 1000000).toFixed(3)} km\u00b2`;
    return `${area.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} m\u00b2`;
  }
  function formatAreaWithError(value, error) {
    const label = formatArea(value);
    const err = Number(error);
    if (!Number.isFinite(err)) return label;
    return `${label} +/- ${formatArea(err)}`;
  }
  function formatPercentWithError(value, error) {
    const pct = fmtNum(value);
    const err = Number(error);
    if (!Number.isFinite(err)) return `${pct} %`;
    return `${pct} % +/- ${fmtNum(err)} %`;
  }
  function fmtSpeed(value) {
    return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(1) : 'N/A';
  }
  function fmtAltDifference(absoluteAlt, relativeAlt) {
    if (absoluteAlt === null || absoluteAlt === undefined || relativeAlt === null || relativeAlt === undefined) return 'N/A';
    return fmtNum(Number(absoluteAlt) - Number(relativeAlt));
  }
  function tooltipLines(photo) {
    return [
      { text: t('file') },
      { text: `  ${photo.name}` },
      { text: '' },
      { text: t('exif') },
      { text: `  ${t('relativeAltitudeM')} : ${fmtNum(photo.alt_m)}`, altitude: true },
      { text: `  ${t('absoluteAltitudeM')} : ${fmtNum(photo.absolute_alt_m)}`, altitude: true },
      { text: `  ${t('absRelAltM')}    : ${fmtAltDifference(photo.absolute_alt_m, photo.alt_m)}`, altitude: true },
      { text: `  ${t('flightPitchDegree')} : ${fmtNum(photo.flight_pitch_deg)}` },
      { text: `  ${t('gimbalPitchDegree')} : ${fmtNum(photo.gimbal_pitch_deg)}` },
      { text: `  ${t('flightRollDegreeLabel')}  : ${fmtNum(photo.flight_roll_deg)}` },
      { text: `  ${t('flightYawDegree')}     : ${fmtNum(photo.flight_yaw_deg)}` },
      { text: `  ${t('gimbalYawDegree')}     : ${fmtNum(photo.gimbal_yaw_deg)}` },
      { text: `  ${t('flightXYZSpeed')}: ${fmtSpeed(photo.flight_x_speed_mps)}/${fmtSpeed(photo.flight_y_speed_mps)}/${fmtSpeed(photo.flight_z_speed_mps)} [m/s]` },
    ];
  }
  function pointInTri(px, py, a, b, c) {
    const v0x = c[0] - a[0], v0y = c[1] - a[1];
    const v1x = b[0] - a[0], v1y = b[1] - a[1];
    const v2x = px - a[0], v2y = py - a[1];
    const dot00 = v0x * v0x + v0y * v0y;
    const dot01 = v0x * v1x + v0y * v1y;
    const dot02 = v0x * v2x + v0y * v2y;
    const dot11 = v1x * v1x + v1y * v1y;
    const dot12 = v1x * v2x + v1y * v2y;
    const denom = dot00 * dot11 - dot01 * dot01;
    if (Math.abs(denom) < 1e-9) return false;
    const inv = 1 / denom;
    const u = (dot11 * dot02 - dot01 * dot12) * inv;
    const v = (dot00 * dot12 - dot01 * dot02) * inv;
    return u >= 0 && v >= 0 && (u + v) <= 1;
  }
  function pointInPhoto(px, py, photo) {
    const c = photo.corners;
    if (!c || c.length < 4) return false;
    return pointInTri(px, py, c[0], c[1], c[2]) || pointInTri(px, py, c[0], c[2], c[3]);
  }
  function bilinearPoint(photo, u, v) {
    const c = photo && photo.corners;
    if (!c || c.length < 4) return [0, 0];
    const a = c[0], b = c[1], d = c[3], q = [
      c[0][0] - c[1][0] + c[2][0] - c[3][0],
      c[0][1] - c[1][1] + c[2][1] - c[3][1],
    ];
    return [
      a[0] + (b[0] - a[0]) * u + (d[0] - a[0]) * v + q[0] * u * v,
      a[1] + (b[1] - a[1]) * u + (d[1] - a[1]) * v + q[1] * u * v,
    ];
  }
  function solveBilinearLocal(photo, px, py) {
    const c = photo && photo.corners;
    if (!c || c.length < 4) return null;
    const a = c[0], b = c[1], d = c[3], q = [
      c[0][0] - c[1][0] + c[2][0] - c[3][0],
      c[0][1] - c[1][1] + c[2][1] - c[3][1],
    ];
    let u = 0.5;
    let v = 0.5;
    for (let i = 0; i < 12; i += 1) {
      const x = a[0] + (b[0] - a[0]) * u + (d[0] - a[0]) * v + q[0] * u * v;
      const y = a[1] + (b[1] - a[1]) * u + (d[1] - a[1]) * v + q[1] * u * v;
      const fx = x - px;
      const fy = y - py;
      if (Math.hypot(fx, fy) < 1e-4) break;
      const j00 = (b[0] - a[0]) + q[0] * v;
      const j01 = (d[0] - a[0]) + q[0] * u;
      const j10 = (b[1] - a[1]) + q[1] * v;
      const j11 = (d[1] - a[1]) + q[1] * u;
      const det = j00 * j11 - j01 * j10;
      if (Math.abs(det) < 1e-9) break;
      u -= (j11 * fx - j01 * fy) / det;
      v -= (-j10 * fx + j00 * fy) / det;
    }
    return { u, v };
  }
  function photoAtPreview(px, py) {
    if (!state || !Array.isArray(state.photos)) return null;
    for (let i = state.photos.length - 1; i >= 0; i -= 1) {
      if (pointInPhoto(px, py, state.photos[i])) return state.photos[i];
    }
    return null;
  }
  function clientToPreview(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    return {
      px: pan[0] + ((clientX - rect.left) * sx) / zoom,
      py: pan[1] + ((clientY - rect.top) * sy) / zoom,
      sx,
      sy,
      screenX: (clientX - rect.left) * sx,
    };
  }
  function previewToRulerPoint(px, py, preferredPhotoId = null) {
    const preferred = preferredPhotoId == null ? null : photoById(preferredPhotoId);
    const photo = preferred || photoAtPreview(px, py);
    if (photo) {
      const local = solveBilinearLocal(photo, px, py);
      if (local) return { photoId: photo.id, u: local.u, v: local.v };
    }
    return { preview: [px, py] };
  }
  function rulerPointToPreview(point) {
    if (!point) return null;
    if (point.photoId != null) {
      const photo = photoById(point.photoId);
      if (photo) return bilinearPoint(photo, Number(point.u || 0), Number(point.v || 0));
    }
    return Array.isArray(point.preview) ? [Number(point.preview[0] || 0), Number(point.preview[1] || 0)] : null;
  }
  function rulerMeters(a, b) {
    const mpp = Number(state && state.full_canvas && state.full_canvas.mpp);
    const scale = Number(state && state.canvas && state.canvas.scale);
    if (!Number.isFinite(mpp) || mpp <= 0 || !Number.isFinite(scale) || scale <= 0) return NaN;
    return Math.hypot(b[0] - a[0], b[1] - a[1]) * (mpp / scale);
  }
  function formatRulerMeters(meters) {
    if (!Number.isFinite(meters)) return 'N/A';
    if (meters >= 1000) return `${(meters / 1000).toFixed(meters >= 10000 ? 2 : 3)} km`;
    if (meters >= 10) return `${meters.toFixed(2)} m`;
    if (meters >= 1) return `${meters.toFixed(3)} m`;
    return `${(meters * 100).toFixed(1)} cm`;
  }
  function screenDistanceToSegment(p, a, b) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy;
    if (len2 <= 1e-9) return Math.hypot(p[0] - a[0], p[1] - a[1]);
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
    return Math.hypot(p[0] - (a[0] + dx * t), p[1] - (a[1] + dy * t));
  }
  function rulerHitAtClient(clientX, clientY) {
    if (!ruler || !ruler.start) return null;
    const p = clientToPreview(clientX, clientY);
    const a = rulerPointToPreview(ruler.start);
    const b = rulerPointToPreview(ruler.end);
    if (!a || !b) return null;
    const ps = [(p.px - pan[0]) * zoom, (p.py - pan[1]) * zoom];
    const as = previewToScreen(a);
    const bs = previewToScreen(b);
    const handleRadius = 11 * Math.max(1, p.sx / Math.max(p.sy, 1));
    if (Math.hypot(ps[0] - as[0], ps[1] - as[1]) <= handleRadius) return { kind: 'start' };
    if (Math.hypot(ps[0] - bs[0], ps[1] - bs[1]) <= handleRadius) return { kind: 'end' };
    if (ruler.complete && screenDistanceToSegment(ps, as, bs) <= 8) return { kind: 'move' };
    return null;
  }
  function setRulerMode(active) {
    rulerMode = !!active;
    rulerModeBtn.classList.toggle('active', rulerMode);
    rulerModeBtn.setAttribute('aria-pressed', rulerMode ? 'true' : 'false');
    rulerModeIndicatorEl.classList.toggle('active', rulerMode);
    canvas.style.cursor = rulerMode ? 'crosshair' : '';
    focusStage();
  }
  function updateRulerButtons() {
    clearRulerBtn.disabled = !ruler;
    rulerModeBtn.classList.toggle('active', rulerMode);
  }
  function setRulerStatus() {
    if (!ruler) {
      setStatus(rulerMode ? t('rulerStart') : t('rulerCleared'));
      return;
    }
    if (!ruler.complete) {
      setStatus(t('rulerEnd'));
      return;
    }
    const a = rulerPointToPreview(ruler.start);
    const b = rulerPointToPreview(ruler.end);
    setStatus(t('rulerValue', { value: formatRulerMeters(a && b ? rulerMeters(a, b) : NaN) }));
  }
  function drawRulerOverlay() {
    if (!overlayCtx || !ruler || !ruler.start) return;
    const a = rulerPointToPreview(ruler.start);
    const b = rulerPointToPreview(ruler.end);
    if (!a || !b) return;
    const as = previewToScreen(a);
    const bs = previewToScreen(b);
    const meters = rulerMeters(a, b);
    overlayCtx.save();
    overlayCtx.lineCap = 'round';
    overlayCtx.lineJoin = 'round';
    overlayCtx.strokeStyle = 'rgba(255,255,255,.96)';
    overlayCtx.lineWidth = 7;
    overlayCtx.beginPath();
    overlayCtx.moveTo(as[0], as[1]);
    overlayCtx.lineTo(bs[0], bs[1]);
    overlayCtx.stroke();
    overlayCtx.strokeStyle = '#f9ab00';
    overlayCtx.lineWidth = 3;
    overlayCtx.beginPath();
    overlayCtx.moveTo(as[0], as[1]);
    overlayCtx.lineTo(bs[0], bs[1]);
    overlayCtx.stroke();
    for (const pt of [as, bs]) {
      overlayCtx.beginPath();
      overlayCtx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
      overlayCtx.fillStyle = 'rgba(255,255,255,.98)';
      overlayCtx.fill();
      overlayCtx.strokeStyle = '#b06000';
      overlayCtx.lineWidth = 2;
      overlayCtx.stroke();
    }
    const mid = [(as[0] + bs[0]) / 2, (as[1] + bs[1]) / 2];
    const dx = bs[0] - as[0];
    const dy = bs[1] - as[1];
    const len = Math.max(1, Math.hypot(dx, dy));
    const nx = -dy / len;
    const ny = dx / len;
    const label = formatRulerMeters(meters);
    overlayCtx.font = '13px system-ui, -apple-system, Segoe UI, sans-serif';
    overlayCtx.textAlign = 'center';
    overlayCtx.textBaseline = 'middle';
    const labelX = mid[0] + nx * 18;
    const labelY = mid[1] + ny * 18;
    const metrics = overlayCtx.measureText(label);
    overlayCtx.fillStyle = 'rgba(255,255,255,.92)';
    overlayCtx.fillRect(labelX - metrics.width / 2 - 6, labelY - 11, metrics.width + 12, 22);
    overlayCtx.fillStyle = '#5f3700';
    overlayCtx.fillText(label, labelX, labelY);
    overlayCtx.restore();
  }
  function hideTooltip() {
    tooltip.style.display = 'none';
    tooltip.classList.remove('message');
  }
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[ch]));
  }
  function formatGb(bytes) {
    const gb = Number(bytes) / (1024 * 1024 * 1024);
    if (!Number.isFinite(gb)) return 'unknown';
    return `${Math.max(0.1, gb).toFixed(1)} GB`;
  }
  function positionTooltip(ev) {
    tooltip.style.display = 'block';
    const pad = 12;
    const stageRect = canvas.getBoundingClientRect();
    let x = ev.clientX + pad;
    let y = ev.clientY + pad;
    if (x + tooltip.offsetWidth + pad > stageRect.right) x = ev.clientX - tooltip.offsetWidth - pad;
    if (y + tooltip.offsetHeight + pad > stageRect.bottom) y = ev.clientY - tooltip.offsetHeight - pad;
    x = Math.max(stageRect.left + pad, Math.min(stageRect.right - tooltip.offsetWidth - pad, x));
    y = Math.max(stageRect.top + pad, Math.min(stageRect.bottom - tooltip.offsetHeight - pad, y));
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
  }
  function showTooltip(ev, text) {
    tooltip.textContent = text;
    tooltip.classList.add('message');
    positionTooltip(ev);
  }
  function showPhotoTooltip(ev, photo) {
    const frag = document.createDocumentFragment();
    const lines = tooltipLines(photo);
    for (let i = 0; i < lines.length; i += 1) {
      const line = document.createElement('span');
      line.textContent = lines[i].text;
      if (lines[i].altitude && photo.altitude_shift_alert) line.className = 'altitude-alert';
      frag.appendChild(line);
      if (i + 1 < lines.length) frag.appendChild(document.createTextNode('\n'));
    }
    tooltip.replaceChildren(frag);
    tooltip.classList.remove('message');
    positionTooltip(ev);
  }
  function chooseOpacitySaveMode(opacity) {
    return new Promise(resolve => {
      opacityDialogTitle.textContent = t('saveWithTransparencyTitle');
      opacityDialogMessage.textContent = t('opacitySaveMessage', { opacity });
      opacityDialogSave.textContent = t('save');
      opacityDialogSaveOpaque.textContent = t('saveWithoutTransparency');
      opacityDialogSaveOpaque.style.display = '';
      opacityDialogCancel.textContent = t('cancel');
      modalBackdrop.classList.add('active');
      function cleanup(result) {
        modalBackdrop.classList.remove('active');
        opacityDialogSave.removeEventListener('click', onSave);
        opacityDialogSaveOpaque.removeEventListener('click', onSaveOpaque);
        opacityDialogCancel.removeEventListener('click', onCancel);
        modalBackdrop.removeEventListener('click', onBackdrop);
        window.removeEventListener('keydown', onKey);
        resolve(result);
      }
      function onSave() { cleanup('save'); }
      function onSaveOpaque() { cleanup('opaque'); }
      function onCancel() { cleanup('cancel'); }
      function onBackdrop(ev) {
        if (ev.target === modalBackdrop) cleanup('cancel');
      }
      function onKey(ev) {
        if (ev.key === 'Escape') cleanup('cancel');
      }
      opacityDialogSave.addEventListener('click', onSave);
      opacityDialogSaveOpaque.addEventListener('click', onSaveOpaque);
      opacityDialogCancel.addEventListener('click', onCancel);
      modalBackdrop.addEventListener('click', onBackdrop);
      window.addEventListener('keydown', onKey);
      opacityDialogSave.focus();
    });
  }
  function chooseMemoryForceSave(shortageBytes) {
    return new Promise(resolve => {
      opacityDialogTitle.textContent = t('insufficientMemoryTitle');
      opacityDialogMessage.textContent = t('memoryForceMessage', { shortage: formatGb(shortageBytes) });
      opacityDialogSave.textContent = t('forceSave');
      opacityDialogSaveOpaque.style.display = 'none';
      opacityDialogCancel.textContent = t('cancel');
      modalBackdrop.classList.add('active');
      function cleanup(result) {
        modalBackdrop.classList.remove('active');
        opacityDialogSaveOpaque.style.display = '';
        opacityDialogSave.removeEventListener('click', onForce);
        opacityDialogCancel.removeEventListener('click', onCancel);
        modalBackdrop.removeEventListener('click', onBackdrop);
        window.removeEventListener('keydown', onKey);
        resolve(result);
      }
      function onForce() { cleanup('force'); }
      function onCancel() { cleanup('cancel'); }
      function onBackdrop(ev) {
        if (ev.target === modalBackdrop) cleanup('cancel');
      }
      function onKey(ev) {
        if (ev.key === 'Escape') cleanup('cancel');
      }
      opacityDialogSave.addEventListener('click', onForce);
      opacityDialogCancel.addEventListener('click', onCancel);
      modalBackdrop.addEventListener('click', onBackdrop);
      window.addEventListener('keydown', onKey);
      opacityDialogSave.focus();
    });
  }
  function showSaveFailureDialog(error) {
    const message = String(error || t('unknownError'));
    if (shownSaveError === message) return;
    shownSaveError = message;
    opacityDialogTitle.textContent = t('saveFailedTitle');
    opacityDialogMessage.textContent = message;
    opacityDialogSave.textContent = t('ok');
    opacityDialogSaveOpaque.style.display = 'none';
    opacityDialogCancel.style.display = 'none';
    modalBackdrop.classList.add('active');
    function cleanup() {
      modalBackdrop.classList.remove('active');
      opacityDialogSaveOpaque.style.display = '';
      opacityDialogCancel.style.display = '';
      opacityDialogSave.removeEventListener('click', cleanup);
      modalBackdrop.removeEventListener('click', onBackdrop);
      window.removeEventListener('keydown', onKey);
    }
    function onBackdrop(ev) {
      if (ev.target === modalBackdrop) cleanup();
    }
    function onKey(ev) {
      if (ev.key === 'Escape' || ev.key === 'Enter') cleanup();
    }
    opacityDialogSave.addEventListener('click', cleanup);
    modalBackdrop.addEventListener('click', onBackdrop);
    window.addEventListener('keydown', onKey);
    opacityDialogSave.focus();
  }
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    if (basemapCanvas.width !== w || basemapCanvas.height !== h) {
      basemapCanvas.width = w;
      basemapCanvas.height = h;
    }
    if (overlayCanvas.width !== w || overlayCanvas.height !== h) {
      overlayCanvas.width = w;
      overlayCanvas.height = h;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    draw();
  }
  function fit() {
    if (!state) return;
    zoom = Math.min(canvas.width / state.canvas.width, canvas.height / state.canvas.height) * 0.96;
    zoom = Math.max(zoom, 0.0001);
    pan = [
      (state.canvas.width - canvas.width / zoom) / 2,
      (state.canvas.height - canvas.height / zoom) / 2,
    ];
    draw();
  }
  function makeTexture(img) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    return tex;
  }
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }
  function loadKmzImage(url, generation, overlayItem) {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        if (generation === kmzLoadGeneration && kmzOverlay && kmzOverlay.items.includes(overlayItem)) {
          overlayItem.image = img;
          draw();
        }
        resolve(img);
      };
      img.onerror = () => {
        overlayItem.failed = true;
        resolve(null);
      };
      img.src = url;
    });
  }
  function posData(c) {
    return new Float32Array([
      c[0][0], c[0][1], c[1][0], c[1][1], c[2][0], c[2][1],
      c[0][0], c[0][1], c[2][0], c[2][1], c[3][0], c[3][1],
    ]);
  }
  async function rebuildLayers() {
    const generation = ++loadGeneration;
    const photosToLoad = [...state.photos];
    layers.forEach(l => gl.deleteTexture(l.texture));
    layers = [];
    setStatus(t('loadingTextures', { count: photosToLoad.length }));
    let loaded = 0;
    for (const p of photosToLoad) {
      if (generation !== loadGeneration) return;
      try {
        const img = await loadImage(p.url);
        if (generation !== loadGeneration) return;
        const latest = photoById(p.id) || p;
        layers.push({ id: p.id, texture: makeTexture(img), pos: posData(latest.corners), name: p.name });
        loaded += 1;
        setStatus(t('loadedTextures', { loaded, total: photosToLoad.length }));
        draw();
      } catch (err) {
        console.warn('Image load failed', p.url, err);
      }
    }
    setStatus(t('readyTextures', { loaded, total: photosToLoad.length }));
  }
  function updateLayerGeometry() {
    const byId = new Map(layers.map(layer => [layer.id, layer]));
    const next = [];
    for (const p of state.photos) {
      const layer = byId.get(p.id);
      if (!layer) continue;
      layer.pos = posData(p.corners);
      next.push(layer);
    }
    layers = next;
    return true;
  }
  function previewToScreen(pt) {
    return [(pt[0] - pan[0]) * zoom, (pt[1] - pan[1]) * zoom];
  }
  function rotatePreviewCorners(corners, degrees) {
    const angle = -Number(degrees || 0) * Math.PI / 180;
    if (!Number.isFinite(angle) || Math.abs(angle) < 1e-9) return corners;
    const cx = corners.reduce((sum, p) => sum + p[0], 0) / corners.length;
    const cy = corners.reduce((sum, p) => sum + p[1], 0) / corners.length;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return corners.map(p => {
      const dx = p[0] - cx;
      const dy = p[1] - cy;
      return [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos];
    });
  }
  function kmzItemPreviewCorners(item) {
    if (!item) return null;
    if (Array.isArray(item.coordinates) && item.coordinates.length >= 4) {
      return item.coordinates.slice(0, 4).map(coord => latLonToPreview(coord.lat, coord.lon));
    }
    const box = item.lat_lon_box;
    if (!box) return null;
    const corners = [
      latLonToPreview(box.north, box.west),
      latLonToPreview(box.north, box.east),
      latLonToPreview(box.south, box.east),
      latLonToPreview(box.south, box.west),
    ];
    return rotatePreviewCorners(corners, Number(box.rotation || 0));
  }
  function drawImageInQuad(ctx, img, corners) {
    if (!img || !corners || corners.length < 4) return;
    const pts = corners.map(previewToScreen);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i += 1) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.clip();
    const w = img.naturalWidth || img.width || 1;
    const h = img.naturalHeight || img.height || 1;
    const ux = [(pts[1][0] - pts[0][0]) / w, (pts[1][1] - pts[0][1]) / w];
    const vx = [(pts[3][0] - pts[0][0]) / h, (pts[3][1] - pts[0][1]) / h];
    ctx.setTransform(ux[0], ux[1], vx[0], vx[1], pts[0][0], pts[0][1]);
    ctx.drawImage(img, 0, 0, w, h);
    ctx.restore();
  }
  function drawKmzOverlay(ctx) {
    if (!kmzOverlay) return;
    ctx.save();
    ctx.globalAlpha = 0.72;
    for (const item of Array.isArray(kmzOverlay.items) ? kmzOverlay.items : []) {
      if (!item.image) continue;
      drawImageInQuad(ctx, item.image, kmzItemPreviewCorners(item));
    }
    ctx.restore();
  }
  function drawKmzMarkers(ctx) {
    if (!kmzOverlay || !Array.isArray(kmzOverlay.markers)) return;
    ctx.save();
    ctx.font = '11px system-ui, -apple-system, Segoe UI, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (const marker of kmzOverlay.markers) {
      const p = previewToScreen(latLonToPreview(marker.lat, marker.lon));
      if (p[0] < -80 || p[1] < -40 || p[0] > overlayCanvas.width + 120 || p[1] > overlayCanvas.height + 40) continue;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = '#d93025';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(255,255,255,.95)';
      ctx.stroke();
      const label = String(marker.name || '').trim();
      if (!label) continue;
      const x = p[0] + 7;
      const y = p[1];
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(255,255,255,.95)';
      ctx.strokeText(label, x, y);
      ctx.fillStyle = '#202124';
      ctx.fillText(label, x, y);
    }
    ctx.restore();
  }
  function latLonToPreview(lat, lon) {
    const g = state && state.georef ? state.georef : null;
    const full = state && state.full_canvas ? state.full_canvas : null;
    const scale = state && state.canvas ? Number(state.canvas.scale || 1) : 1;
    if (!g || !full) return [0, 0];
    const x = (Number(lon) - Number(g.origin_lon)) * Number(g.m_per_deg_lon);
    const y = (Number(lat) - Number(g.origin_lat)) * Number(g.m_per_deg_lat);
    const u = (x - Number(g.min_x)) / Number(full.mpp) * scale;
    const v = (Number(g.max_y) - y) / Number(full.mpp) * scale;
    return [u, v];
  }
  function previewToLatLon(u, v) {
    const g = state && state.georef ? state.georef : null;
    const full = state && state.full_canvas ? state.full_canvas : null;
    const scale = state && state.canvas ? Number(state.canvas.scale || 1) : 1;
    if (!g || !full) return [0, 0];
    const x = Number(g.min_x) + (Number(u) / scale) * Number(full.mpp);
    const y = Number(g.max_y) - (Number(v) / scale) * Number(full.mpp);
    const lat = Number(g.origin_lat) + y / Number(g.m_per_deg_lat);
    const lon = Number(g.origin_lon) + x / Number(g.m_per_deg_lon);
    return [lat, lon];
  }
  function lonToTileX(lon, z) {
    return Math.floor(((Number(lon) + 180) / 360) * Math.pow(2, z));
  }
  function latToTileY(lat, z) {
    const rad = Number(lat) * Math.PI / 180;
    return Math.floor((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2 * Math.pow(2, z));
  }
  function tileXToLon(x, z) {
    return x / Math.pow(2, z) * 360 - 180;
  }
  function tileYToLat(y, z) {
    const n = Math.PI - 2 * Math.PI * y / Math.pow(2, z);
    return 180 / Math.PI * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
  }
  function chooseTileZoom(provider) {
    const full = state && state.full_canvas ? state.full_canvas : null;
    const scale = state && state.canvas ? Number(state.canvas.scale || 1) : 1;
    const g = state && state.georef ? state.georef : null;
    if (!full || !g) return provider === 'gsi' ? 16 : 17;
    const metersPerScreenPx = (Number(full.mpp) / scale) / Math.max(zoom, 0.0001);
    const lat = Number(g.origin_lat);
    const earth = 40075016.686;
    const raw = Math.log2(Math.cos(lat * Math.PI / 180) * earth / (256 * Math.max(metersPerScreenPx, 0.01)));
    const maxZ = provider === 'gsi' ? 18 : 19;
    return Math.max(1, Math.min(maxZ, Math.floor(raw)));
  }
  function loadTile(provider, z, x, y) {
    const max = Math.pow(2, z);
    if (x < 0 || y < 0 || x >= max || y >= max) return null;
    const key = `${provider}/${z}/${x}/${y}`;
    let item = tileCache.get(key);
    if (item) return item;
    const img = new Image();
    item = { img, loaded: false, failed: false };
    img.onload = () => { item.loaded = true; draw(); };
    img.onerror = () => { item.failed = true; };
    img.src = `/tile/${provider}/${z}/${x}/${y}`;
    tileCache.set(key, item);
    return item;
  }
  function visibleBasemapTiles() {
    const result = [];
    if (!state || !basemapEnabled()) return result;
    const provider = basemapProvider();
    let z = chooseTileZoom(provider);
    const visible = [
      previewToLatLon(pan[0], pan[1]),
      previewToLatLon(pan[0] + canvas.width / Math.max(zoom, 0.0001), pan[1] + canvas.height / Math.max(zoom, 0.0001)),
    ];
    const minLat = Math.max(-85.0511, Math.min(visible[0][0], visible[1][0]));
    const maxLat = Math.min(85.0511, Math.max(visible[0][0], visible[1][0]));
    const minLon = Math.max(-180, Math.min(visible[0][1], visible[1][1]));
    const maxLon = Math.min(180, Math.max(visible[0][1], visible[1][1]));
    let x0 = lonToTileX(minLon, z);
    let x1 = lonToTileX(maxLon, z);
    let y0 = latToTileY(maxLat, z);
    let y1 = latToTileY(minLat, z);
    while ((x1 - x0 + 1) * (y1 - y0 + 1) > 180 && z > 1) {
      z -= 1;
      x0 = lonToTileX(minLon, z);
      x1 = lonToTileX(maxLon, z);
      y0 = latToTileY(maxLat, z);
      y1 = latToTileY(minLat, z);
    }
    for (let ty = y0; ty <= y1; ty += 1) {
      for (let tx = x0; tx <= x1; tx += 1) {
        const tile = loadTile(provider, z, tx, ty);
        if (!tile || !tile.loaded) continue;
        const nw = latLonToPreview(tileYToLat(ty, z), tileXToLon(tx, z));
        const se = latLonToPreview(tileYToLat(ty + 1, z), tileXToLon(tx + 1, z));
        result.push({ tile, corners: [[nw[0], nw[1]], [se[0], nw[1]], [se[0], se[1]], [nw[0], se[1]]] });
      }
    }
    return result;
  }
  function drawBasemapGl() {
    if (!state || !basemapEnabled()) return;
    gl.uniform1f(loc.opacity, 1);
    gl.uniform1i(loc.compareMode, 0);
    gl.uniform1f(loc.layerParity, 0);
    gl.uniform1f(loc.swipeX, 0);
    for (const entry of visibleBasemapTiles()) {
      if (!entry.tile.texture) entry.tile.texture = makeTexture(entry.tile.img);
      gl.bindTexture(gl.TEXTURE_2D, entry.tile.texture);
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, posData(entry.corners), gl.STATIC_DRAW);
      gl.vertexAttribPointer(loc.pos, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }
  function drawMapAttribution(ctx) {
    if (!basemapEnabled()) return;
    const provider = basemapProvider();
    const text = provider === 'gsi' ? t('sourceGsi') : '© OpenStreetMap contributors';
    ctx.save();
    ctx.font = '12px system-ui, -apple-system, Segoe UI, sans-serif';
    const pad = 6;
    const w = ctx.measureText(text).width + pad * 2;
    const h = 22;
    const x = overlayCanvas.width - w - 8;
    const y = overlayCanvas.height - h - 8;
    ctx.fillStyle = 'rgba(255,255,255,.86)';
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = '#202124';
    ctx.fillText(text, x + pad, y + 15);
    ctx.restore();
  }
  function compareMode() {
    const checked = compareModeEls.find(el => el.checked);
    return checked ? checked.value : 'normal';
  }
  function pairModeActive(mode) {
    return mode === 'pair-swipe' || mode === 'pair-red-cyan';
  }
  function pairModeShader(mode) {
    return mode === 'pair-red-cyan' ? 2 : 1;
  }
  function layerById(id) {
    return id == null ? null : layers.find(layer => layer.id === id) || null;
  }
  function drawPhotoLayer(layer, opacity, shaderMode = 0, parity = 0) {
    gl.uniform1f(loc.opacity, opacity);
    gl.uniform1i(loc.compareMode, shaderMode);
    gl.uniform1f(loc.layerParity, parity);
    gl.bindTexture(gl.TEXTURE_2D, layer.texture);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, layer.pos, gl.STATIC_DRAW);
    gl.vertexAttribPointer(loc.pos, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
  function drawLayerOutlines(ctx) {
    const mode = compareMode();
    if (!pairModeActive(mode) || !layers.length) return;
    ctx.save();
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.setLineDash([]);
    for (let i = 0; i < layers.length; i += 1) {
      if (pairModeActive(mode) && layers[i].id !== hoverPhotoId && layers[i].id !== hoverUnderPhotoId) continue;
      const p = layers[i].pos;
      if (!p || p.length < 12) continue;
      const corners = [
        previewToScreen([p[0], p[1]]),
        previewToScreen([p[2], p[3]]),
        previewToScreen([p[4], p[5]]),
        previewToScreen([p[10], p[11]]),
      ];
      const color = i % 2 ? '#e8710a' : '#1a73e8';
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let j = 1; j < corners.length; j += 1) ctx.lineTo(corners[j][0], corners[j][1]);
      ctx.closePath();
      ctx.strokeStyle = 'rgba(255,255,255,.92)';
      ctx.lineWidth = 5;
      ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }
  function drawArrow(ctx, from, to) {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const len = Math.hypot(dx, dy);
    if (len < 1) return;
    const ux = dx / len;
    const uy = dy / len;
    const start = [from[0] + ux * 11, from[1] + uy * 11];
    const end = [to[0] - ux * 13, to[1] - uy * 13];
    ctx.beginPath();
    ctx.moveTo(start[0], start[1]);
    ctx.lineTo(end[0], end[1]);
    ctx.stroke();
    const size = 12;
    ctx.beginPath();
    ctx.moveTo(end[0], end[1]);
    ctx.lineTo(end[0] - ux * size - uy * size * 0.55, end[1] - uy * size + ux * size * 0.55);
    ctx.lineTo(end[0] - ux * size + uy * size * 0.55, end[1] - uy * size - ux * size * 0.55);
    ctx.closePath();
    ctx.fill();
  }
  function drawSequenceOverlay() {
    if (!overlayCtx) return;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    drawKmzOverlay(overlayCtx);
    drawKmzMarkers(overlayCtx);
    drawMapAttribution(overlayCtx);
    drawLayerOutlines(overlayCtx);
    if (!state || !captureOrderEnabled()) {
      drawRulerOverlay();
      return;
    }
    const seq = Array.isArray(state.sequence) ? state.sequence : [];
    const pts = seq.map(item => ({ ...item, screen: previewToScreen([item.center[0], item.center[1]]) }));
    overlayCtx.save();
    overlayCtx.lineCap = 'round';
    overlayCtx.lineJoin = 'round';
    for (let i = 0; i < pts.length - 1; i += 1) {
      const isAltitudeTransition = !!pts[i + 1].altitude_shift_from_previous;
      overlayCtx.lineWidth = isAltitudeTransition ? 8 : 5;
      overlayCtx.strokeStyle = isAltitudeTransition ? '#d93025' : '#1a73e8';
      overlayCtx.fillStyle = isAltitudeTransition ? '#d93025' : '#1a73e8';
      drawArrow(overlayCtx, pts[i].screen, pts[i + 1].screen);
    }
    overlayCtx.font = '12px system-ui, -apple-system, Segoe UI, sans-serif';
    overlayCtx.textAlign = 'center';
    overlayCtx.textBaseline = 'top';
    for (const p of pts) {
      overlayCtx.beginPath();
      overlayCtx.arc(p.screen[0], p.screen[1], 6, 0, Math.PI * 2);
      overlayCtx.fillStyle = '#1a73e8';
      overlayCtx.fill();
      const label = p.captured_at || p.name || '';
      if (label) {
        const textY = p.screen[1] + 10;
        const metrics = overlayCtx.measureText(label);
        overlayCtx.fillStyle = 'rgba(255,255,255,.84)';
        overlayCtx.fillRect(p.screen[0] - metrics.width / 2 - 4, textY - 1, metrics.width + 8, 16);
        overlayCtx.fillStyle = '#174ea6';
        overlayCtx.fillText(label, p.screen[0], textY);
      }
    }
    overlayCtx.restore();
    drawRulerOverlay();
  }
  function altitudeShiftArrowHitAtClient(clientX, clientY) {
    if (!state || !captureOrderEnabled() || !Array.isArray(state.sequence)) return false;
    const rect = canvas.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return false;
    const p = [
      (clientX - rect.left) * (canvas.width / rect.width),
      (clientY - rect.top) * (canvas.height / rect.height),
    ];
    const pts = state.sequence.map(item => ({ ...item, screen: previewToScreen([item.center[0], item.center[1]]) }));
    for (let i = 0; i < pts.length - 1; i += 1) {
      if (pts[i + 1].altitude_shift_from_previous && screenDistanceToSegment(p, pts[i].screen, pts[i + 1].screen) <= 10) {
        return true;
      }
    }
    return false;
  }
  function photoLayerOpacity() {
    return Math.max(0, Math.min(1, (Number(state.options.opacity_pct) || 100) / 100));
  }
  function niceScaleMeters(rawMeters) {
    if (!Number.isFinite(rawMeters) || rawMeters <= 0) return 0;
    const exp = Math.floor(Math.log10(rawMeters));
    const base = rawMeters / Math.pow(10, exp);
    const niceBase = base >= 5 ? 5 : base >= 2 ? 2 : 1;
    return niceBase * Math.pow(10, exp);
  }
  function formatScaleMeters(meters) {
    if (meters >= 1000) {
      const km = meters / 1000;
      return `${Number.isInteger(km) ? km.toFixed(0) : km.toFixed(1)} km`;
    }
    if (meters < 1) {
      const cm = meters * 100;
      return `${Number.isInteger(cm) ? cm.toFixed(0) : cm.toFixed(1)} cm`;
    }
    if (!Number.isInteger(meters)) {
      return `${meters.toFixed(1).replace(/\.0$/, '')} m`;
    }
    return `${Math.round(meters)} m`;
  }
  function isNearMultiple(value, interval) {
    if (!Number.isFinite(value) || !Number.isFinite(interval) || interval <= 0) return false;
    return Math.abs(value / interval - Math.round(value / interval)) < 1e-6;
  }
  function scaleTickIntervals(meters) {
    const exp = Math.floor(Math.log10(meters));
    const pow = Math.pow(10, exp);
    const base = Math.round((meters / pow) * 10) / 10;
    const major = Math.pow(10, base === 1 ? exp - 1 : exp);
    const strong = base === 1 ? meters / 2 : 0;
    return { major, strong };
  }
  function renderScaleTicks(scaleBar, meters, vertical = false) {
    if (!scaleBar || !Number.isFinite(meters) || meters <= 0) return;
    const tickStep = meters / 20;
    const intervals = scaleTickIntervals(meters);
    const frag = document.createDocumentFragment();
    for (let i = 1; i < 20; i++) {
      const tickValue = tickStep * i;
      const tick = document.createElement('span');
      tick.className = 'scale-tick';
      if (intervals.strong && isNearMultiple(tickValue, intervals.strong)) {
        tick.classList.add('strong');
      } else if (isNearMultiple(tickValue, intervals.major)) {
        tick.classList.add('major');
      }
      if (vertical) {
        tick.style.bottom = `${(i / 20) * 100}%`;
      } else {
        tick.style.left = `${(i / 20) * 100}%`;
      }
      frag.appendChild(tick);
    }
    scaleBar.replaceChildren(frag);
  }
  function renderScaleLabels(labelEl, meters, widthPx) {
    if (!labelEl || !Number.isFinite(meters) || meters <= 0) return;
    labelEl.style.width = `${widthPx.toFixed(0)}px`;
    labelEl.replaceChildren();
    const midLabel = document.createElement('span');
    midLabel.className = 'scale-label-mid';
    midLabel.textContent = formatScaleMeters(meters / 2);
    labelEl.appendChild(midLabel);
    const endLabel = document.createElement('span');
    endLabel.className = 'scale-label-end';
    endLabel.textContent = formatScaleMeters(meters);
    labelEl.appendChild(endLabel);
  }
  function renderVerticalScaleLabels(labelEl, meters, heightPx) {
    if (!labelEl || !Number.isFinite(meters) || meters <= 0) return;
    labelEl.style.height = `${heightPx.toFixed(0)}px`;
    labelEl.replaceChildren();
    const midLabel = document.createElement('span');
    midLabel.className = 'scale-label-mid';
    midLabel.textContent = formatScaleMeters(meters / 2);
    labelEl.appendChild(midLabel);
    const endLabel = document.createElement('span');
    endLabel.className = 'scale-label-end';
    endLabel.textContent = formatScaleMeters(meters);
    labelEl.appendChild(endLabel);
  }
  function renderScaleIndicator(indicator, meters, widthPx) {
    indicator.style.setProperty('--scale-len', `${widthPx.toFixed(0)}px`);
    indicator.style.width = `${(widthPx + 47).toFixed(0)}px`;
    indicator.style.height = `${(widthPx + 25).toFixed(0)}px`;
    renderScaleLabels(indicator.querySelector('.scale-label'), meters, widthPx);
    const scaleBar = indicator.querySelector('.scale-bar');
    scaleBar.style.width = `${widthPx.toFixed(0)}px`;
    renderScaleTicks(scaleBar, meters);
    renderVerticalScaleLabels(indicator.querySelector('.scale-v-label'), meters, widthPx);
    const verticalScaleBar = indicator.querySelector('.scale-v-bar');
    verticalScaleBar.style.height = `${widthPx.toFixed(0)}px`;
    renderScaleTicks(verticalScaleBar, meters, true);
    indicator.style.display = 'block';
  }
  function updateScaleIndicator() {
    if (!scaleIndicator || !state || !state.full_canvas || !state.canvas) return;
    const full = state.full_canvas;
    const scale = Number(state.canvas.scale || 1);
    const mpp = Number(full.mpp);
    const dpr = canvas.width / Math.max(1, canvas.clientWidth || canvas.width);
    if (!Number.isFinite(mpp) || !Number.isFinite(scale) || scale <= 0 || zoom <= 0) {
      scaleIndicator.style.display = 'none';
      return;
    }
    const metersPerCssPx = (mpp / scale) * dpr / Math.max(zoom, 0.0001);
    const meters = niceScaleMeters(metersPerCssPx * 480);
    const widthPx = Math.max(120, Math.min(720, meters / metersPerCssPx));
    if (!meters || !Number.isFinite(widthPx)) {
      scaleIndicator.style.display = 'none';
      return;
    }
    renderScaleIndicator(scaleIndicator, meters, widthPx);
  }
  function draw() {
    if (!state) return;
    const photoOpacity = photoLayerOpacity();
    const mode = compareMode();
    const useBasemap = basemapEnabled();
    if (basemapCtx) basemapCtx.clearRect(0, 0, basemapCanvas.width, basemapCanvas.height);
    if (useBasemap) {
      gl.clearColor(1, 1, 1, 1);
    } else {
      gl.clearColor(0, 0, 0, 1);
    }
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(program);
    gl.uniform2f(loc.canvas, state.canvas.width, state.canvas.height);
    gl.uniform2f(loc.view, canvas.width, canvas.height);
    gl.uniform1f(loc.zoom, zoom);
    gl.uniform2f(loc.pan, pan[0], pan[1]);
    gl.uniform1f(loc.opacity, photoOpacity);
    gl.uniform1i(loc.compareMode, 0);
    gl.uniform1f(loc.layerParity, 0);
    gl.uniform1f(loc.swipeX, hoverScreenX || canvas.width * 0.5);

    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.enableVertexAttribArray(loc.uv);
    gl.vertexAttribPointer(loc.uv, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc.pos);
    drawBasemapGl();
    gl.uniform1f(loc.opacity, photoOpacity);
    gl.uniform1i(loc.compareMode, 0);
    gl.uniform1f(loc.layerParity, 0);
    gl.uniform1f(loc.swipeX, hoverScreenX || canvas.width * 0.5);
    const focusedLayer = pairModeActive(mode) ? layerById(hoverPhotoId) : null;
    const underLayer = pairModeActive(mode) ? layerById(hoverUnderPhotoId) : null;
    if (pairModeActive(mode) && focusedLayer && underLayer) {
      const shaderMode = pairModeShader(mode);
      if (shaderMode === 2) gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      drawPhotoLayer(underLayer, 1, shaderMode, 0);
      drawPhotoLayer(focusedLayer, 1, shaderMode, 1);
    } else {
      for (let i = 0; i < layers.length; i += 1) {
        const layer = layers[i];
        if (focusedLayer && layer.id === focusedLayer.id) continue;
        drawPhotoLayer(layer, photoOpacity);
      }
    }
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const previewW = Math.round(state.canvas.width);
    const previewH = Math.round(state.canvas.height);
    const full = state.full_canvas || {};
    const saveW = Math.round(full.width || 0);
    const saveH = Math.round(full.height || 0);
    const mem = full.estimated_memory || 'N/A';
    const memTotal = memoryInfo && memoryInfo.total ? memoryInfo.total : 'N/A';
    const memAvailable = memoryInfo && memoryInfo.available ? memoryInfo.available : 'N/A';
    const memLabel = memoryInfo && memoryInfo.is_wsl ? ' (WSL)' : '';
    const memPct = memoryInfo && Number.isFinite(Number(memoryInfo.available_pct)) ? Math.max(0, Math.min(100, Number(memoryInfo.available_pct))) : 0;
    const lines = `bvpp WebUI\n${state.photos.length} ${t('photos')}\n${t('preview')} ${previewW} x ${previewH} px\n${t('saveSize')} ${saveW} x ${saveH} px\n${t('estimatedMemoryRequired')} ${mem}\n${t('zoom')} ${(zoom * 100).toFixed(1)}%`;
    hud.innerHTML =
      `<div class="hud-lines">${escapeHtml(lines)}</div>` +
      `<div class="mem-row">${escapeHtml(t('physicalMemory'))}${memLabel} ${escapeHtml(memTotal)}<br>${escapeHtml(t('availableMemory'))}${memLabel} ${escapeHtml(memAvailable)}</div>` +
      `<div class="mem-bar"><div class="mem-fill" style="width:${memPct.toFixed(1)}%"></div></div>`;
    updateScaleIndicator();
    drawSequenceOverlay();
  }
  async function updateMemoryInfo() {
    try {
      const res = await fetch('/api/memory');
      memoryInfo = await res.json();
      draw();
    } catch (err) {
      memoryInfo = null;
      draw();
    }
  }
  function setMemoryPolling(intervalMs) {
    if (memoryPollInterval === intervalMs && memoryPollTimer) return;
    if (memoryPollTimer) clearInterval(memoryPollTimer);
    memoryPollInterval = intervalMs;
    memoryPollTimer = setInterval(updateMemoryInfo, intervalMs);
  }
  function isFocused(el) {
    return document.activeElement === el;
  }
  function syncControlsFromState() {
    if (!isFocused(altEl)) altEl.value = state.options.alt_correction_m;
    if (!isFocused(yawEl)) yawEl.value = state.options.yaw_offset_deg;
    yawModeEl.value = state.options.yaw_mode;
    yawInvertEl.checked = !!state.options.yaw_invert;
    opacityEl.value = state.options.opacity_pct;
    opacityTextEl.textContent = `${Math.round(state.options.opacity_pct)}%`;
    cropOptimizeEl.checked = !!state.options.crop_optimize;
    if (!isFocused(altEl) && !isFocused(yawEl)) {
      lastCommittedNumberText = `${altEl.value}|${yawEl.value}`;
    }
    updateInfo();
  }
  async function loadState(refit = false) {
    const res = await fetch('/api/state');
    state = await res.json();
    syncControlsFromState();
    if (refit) fit();
    await rebuildLayers();
    if (refit) fit();
    draw();
  }
  let optionTimer = null;
  let optionChangeGeneration = 0;
  let optionUpdateInFlight = false;
  let queuedOptionReloadTextures = false;
  async function updateOptions(reloadTextures = false) {
    if (optionUpdateInFlight) {
      queuedOptionReloadTextures = queuedOptionReloadTextures || reloadTextures;
      return;
    }
    optionUpdateInFlight = true;
    const requestGeneration = optionChangeGeneration;
    opacityTextEl.textContent = `${opacityEl.value}%`;
    const payload = {
      alt_correction_m: parseFloat(altEl.value || '0'),
      yaw_offset_deg: parseFloat(yawEl.value || '0'),
      yaw_mode: yawModeEl.value,
      yaw_invert: yawInvertEl.checked,
      opacity_pct: parseFloat(opacityEl.value || '100'),
      crop_optimize: cropOptimizeEl.checked,
    };
    setStatus(t('updatingGeometry'));
    try {
      const res = await fetch('/api/options', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      const nextState = await res.json();
      if (requestGeneration !== optionChangeGeneration) return;
      state = nextState;
      syncControlsFromState();
      if (reloadTextures || !updateLayerGeometry()) {
        await rebuildLayers();
      } else {
        setStatus(t('readyGeometryUpdated'));
      }
      draw();
      scheduleAreaStats();
    } finally {
      optionUpdateInFlight = false;
      if (requestGeneration !== optionChangeGeneration || queuedOptionReloadTextures) {
        const queuedReloadTextures = queuedOptionReloadTextures;
        queuedOptionReloadTextures = false;
        clearTimeout(optionTimer);
        optionTimer = null;
        updateOptions(queuedReloadTextures).catch(err => setStatus(t('updateFailed', { error: err })));
      }
    }
  }
  function scheduleOptions(reloadTextures = false) {
    optionChangeGeneration += 1;
    invalidateAreaStats();
    clearTimeout(optionTimer);
    optionTimer = setTimeout(() => updateOptions(reloadTextures), 160);
  }
  function numericValue(el, fallback = 0) {
    const value = parseFloat(el.value);
    return Number.isFinite(value) ? value : fallback;
  }
  function setNumericValue(el, value) {
    el.value = Number(value).toFixed(2);
  }
  function commitNumberEdits() {
    const alt = parseFloat(altEl.value);
    const yaw = parseFloat(yawEl.value);
    if (!Number.isFinite(alt)) altEl.value = state ? state.options.alt_correction_m : '0';
    if (!Number.isFinite(yaw)) yawEl.value = state ? state.options.yaw_offset_deg : '0';
    const currentText = `${altEl.value}|${yawEl.value}`;
    if (currentText === lastCommittedNumberText) return Promise.resolve();
    lastCommittedNumberText = currentText;
    optionChangeGeneration += 1;
    invalidateAreaStats();
    clearTimeout(optionTimer);
    return updateOptions(false);
  }
  function onNumberEditKeydown(ev) {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      commitNumberEdits();
      focusStage();
    } else if (ev.key === 'Escape') {
      ev.preventDefault();
      if (state) {
        ev.currentTarget.value = ev.currentTarget === altEl ? state.options.alt_correction_m : state.options.yaw_offset_deg;
      }
      focusStage();
    }
  }
  function focusStage() {
    canvas.focus({ preventScroll: true });
  }
  function showTab(name) {
    tabButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === name));
    tabPanels.forEach(panel => panel.classList.toggle('active', panel.dataset.panel === name));
    if (name === 'area') scheduleAreaStats();
    else invalidateAreaStats();
    resize();
  }
  function isTypingTarget(target) {
    if (!target) return false;
    const tag = String(target.tagName || '').toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable;
  }
  window.addEventListener('keydown', (ev) => {
    if (ev.defaultPrevented || ev.altKey || ev.ctrlKey || ev.metaKey || isTypingTarget(ev.target)) return;
    const key = ev.key.toLowerCase();
    if (key === 'h') {
      setNumericValue(altEl, numericValue(altEl) + 0.2);
    } else if (key === 'j') {
      setNumericValue(altEl, numericValue(altEl) - 0.2);
    } else if (key === 'k') {
      setNumericValue(yawEl, numericValue(yawEl) + 2.0);
    } else if (key === 'l') {
      setNumericValue(yawEl, numericValue(yawEl) - 2.0);
    } else {
      return;
    }
    ev.preventDefault();
    scheduleOptions(false);
  });
  [altEl, yawEl].forEach(el => {
    el.addEventListener('change', commitNumberEdits);
    el.addEventListener('keydown', onNumberEditKeydown);
  });
  [yawModeEl, yawInvertEl, cropOptimizeEl].forEach(el => el.addEventListener('input', () => scheduleOptions(false)));
  opacityEl.addEventListener('input', () => {
    opacityTextEl.textContent = `${opacityEl.value}%`;
    if (state) state.options.opacity_pct = parseFloat(opacityEl.value || '100');
    draw();
    scheduleOptions(false);
  });
  opacityEl.addEventListener('change', focusStage);
  opacityEl.addEventListener('pointerup', focusStage);
  yawModeEl.addEventListener('change', focusStage);
  yawInvertEl.addEventListener('change', focusStage);
  cropOptimizeEl.addEventListener('change', focusStage);
  compareModeEls.forEach(el => el.addEventListener('change', () => {
    draw();
    focusStage();
  }));
  showCaptureOrderEl.addEventListener('change', draw);
  showBasemapEl.addEventListener('change', () => {
    draw();
  });
  basemapProviderEls.forEach(el => el.addEventListener('change', draw));
  async function loadKmzOverlay(file) {
    if (!file) return;
    const generation = ++kmzLoadGeneration;
    kmzOverlay = null;
    clearKmzBtn.disabled = true;
    setLines(kmzInfoEl, t('loadingKmz'));
    draw();
    try {
      const res = await fetch(`/api/kmz-overlay?name=${encodeURIComponent(file.name || 'overlay.kmz')}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/vnd.google-earth.kmz'},
        body: await file.arrayBuffer(),
      });
      const payload = await res.json();
      if (!res.ok || !payload.ok) throw new Error(payload.error || res.statusText);
      if (generation !== kmzLoadGeneration) return;
      const items = Array.isArray(payload.items) ? payload.items : [];
      const markers = Array.isArray(payload.markers) ? payload.markers : [];
      const featureCount = items.length + markers.length;
      if (!featureCount) {
        setLines(kmzInfoEl, t('kmzNoOverlay'));
        return;
      }
      kmzOverlay = { name: payload.name || file.name, items, markers };
      clearKmzBtn.disabled = false;
      setLines(kmzInfoEl, t('kmzLoaded', { name: kmzOverlay.name, count: featureCount }));
      items.forEach(item => loadKmzImage(item.data_url, generation, item));
      draw();
    } catch (err) {
      if (generation !== kmzLoadGeneration) return;
      kmzOverlay = null;
      clearKmzBtn.disabled = true;
      setLines(kmzInfoEl, t('kmzLoadFailed', { error: err && err.message ? err.message : err }));
      draw();
    } finally {
      if (kmzFileEl) kmzFileEl.value = '';
    }
  }
  kmzFileEl.addEventListener('change', () => {
    const file = kmzFileEl.files && kmzFileEl.files[0] ? kmzFileEl.files[0] : null;
    loadKmzOverlay(file);
  });
  clearKmzBtn.addEventListener('click', () => {
    kmzLoadGeneration += 1;
    kmzOverlay = null;
    clearKmzBtn.disabled = true;
    if (kmzFileEl) kmzFileEl.value = '';
    setLines(kmzInfoEl, t('kmzCleared'));
    draw();
    focusStage();
  });
  tabButtons.forEach(btn => btn.addEventListener('click', () => showTab(btn.dataset.tab)));
  document.getElementById('fit').addEventListener('click', fit);
  rulerModeBtn.addEventListener('click', () => {
    setRulerMode(!rulerMode);
    if (rulerMode) setRulerStatus();
  });
  clearRulerBtn.addEventListener('click', () => {
    ruler = null;
    rulerDrag = null;
    updateRulerButtons();
    setRulerStatus();
    draw();
    focusStage();
  });
  document.getElementById('revert').addEventListener('click', async () => {
    setStatus(t('reverting'));
    const res = await fetch('/api/revert', { method: 'POST' });
    state = await res.json();
    syncControlsFromState();
    await rebuildLayers();
    scheduleAreaStats();
    fit();
    draw();
    setStatus(t('reverted'));
  });
  function renderSaveStatus(payload) {
    if (!payload || !payload.active) {
      if (payload && payload.error) {
        setStatus(t('saveFailed', { error: payload.error }));
        showSaveFailureDialog(payload.error);
      }
      setSaveActive(false);
      return false;
    }
    if (payload.status === 'running') {
      setSaveActive(true, true);
      const pct = payload.total ? Math.round((payload.done / payload.total) * 100) : 0;
      const filePart = payload.current ? `: ${payload.current}` : '';
      setStatus(`${translatePhase(payload.phase)} ${pct}% (${payload.done}/${payload.total})${filePart}`);
      return true;
    }
    if (payload.status === 'saving') {
      setSaveActive(true, false);
      setStatus(translateSaveText(payload.save_text));
      return true;
    }
    if (payload.status === 'done') {
      setSaveActive(false);
      setStatus(t('saved', { path: payload.path }));
      return false;
    }
    if (payload.status === 'cancelled') {
      setSaveActive(false);
      setStatus(t('saveCancelled'));
      return false;
    }
    setSaveActive(false);
    const error = payload.error || t('unknownError');
    setStatus(t('saveFailed', { error }));
    showSaveFailureDialog(error);
    return false;
  }
  async function pollSaveStatus() {
    try {
      const res = await fetch('/api/save-status');
      const payload = await res.json();
      const keepPolling = renderSaveStatus(payload);
      if (!keepPolling && savePollTimer) {
        clearInterval(savePollTimer);
        savePollTimer = null;
      }
    } catch (err) {
      setStatus(t('saveStatusFailed', { error: err }));
      setSaveActive(false);
      if (savePollTimer) {
        clearInterval(savePollTimer);
        savePollTimer = null;
      }
    }
  }
  document.getElementById('save').addEventListener('click', async () => {
    let forceSave = false;
    shownSaveError = null;
    await updateMemoryInfo();
    const requiredBytes = Number(state && state.full_canvas ? state.full_canvas.estimated_memory_bytes : NaN);
    const marginBytes = Number(state && state.full_canvas ? state.full_canvas.memory_safety_margin_bytes : 0);
    const availableBytes = Number(memoryInfo ? memoryInfo.available_bytes : NaN);
    const shortageBytes = requiredBytes + marginBytes - availableBytes;
    if (Number.isFinite(shortageBytes) && shortageBytes > 0) {
      const choice = await chooseMemoryForceSave(shortageBytes);
      if (choice === 'cancel') return;
      forceSave = true;
    }
    const opacity = parseFloat(opacityEl.value || '100');
    if (Number.isFinite(opacity) && opacity < 100) {
      const choice = await chooseOpacitySaveMode(opacity);
      if (choice === 'cancel') return;
      if (choice === 'opaque') {
        opacityEl.value = '100';
        opacityTextEl.textContent = '100%';
        if (state) state.options.opacity_pct = 100;
        draw();
        await updateOptions(false);
      }
    }
    setStatus(t('startingSave'));
    setSaveActive(true);
    try {
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({force: forceSave}),
      });
      const payload = await res.json();
      const keepPolling = renderSaveStatus(payload);
      if (keepPolling && !savePollTimer) {
        savePollTimer = setInterval(pollSaveStatus, 500);
      }
      if (keepPolling) pollSaveStatus();
    } catch (err) {
      setStatus(t('saveStartFailed', { error: err }));
      setSaveActive(false);
    }
  });
  cancelSaveBtn.addEventListener('click', async () => {
    setStatus(t('cancellingSave'));
    await fetch('/api/cancel-save', { method: 'POST' });
    pollSaveStatus();
  });
  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const oldZoom = zoom;
    const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoom = Math.min(64, Math.max(0.0001, zoom * factor));
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
    const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
    pan[0] += x / oldZoom - x / zoom;
    pan[1] += y / oldZoom - y / zoom;
    draw();
  }, { passive: false });
  canvas.addEventListener('mousedown', ev => {
    mouseDragged = false;
    mouseDownPos = [ev.clientX, ev.clientY];
    if (rulerMode) {
      hideTooltip();
      const hit = rulerHitAtClient(ev.clientX, ev.clientY);
      if (hit && ruler && ruler.complete) {
        const p = clientToPreview(ev.clientX, ev.clientY);
        rulerDrag = {
          kind: hit.kind,
          origin: [p.px, p.py],
          startPreview: rulerPointToPreview(ruler.start),
          endPreview: rulerPointToPreview(ruler.end),
          startPhotoId: ruler.start && ruler.start.photoId,
          endPhotoId: ruler.end && ruler.end.photoId,
        };
        rulerSuppressClick = false;
        dragging = false;
      } else {
        dragging = true;
        lastMouse = [ev.clientX, ev.clientY];
        canvas.style.cursor = 'grabbing';
      }
      focusStage();
      ev.preventDefault();
      return;
    }
    dragging = true;
    lastMouse = [ev.clientX, ev.clientY];
  });
  window.addEventListener('mouseup', ev => {
    dragging = false;
    if (rulerDrag) {
      rulerDrag = null;
      setRulerStatus();
    }
    if (rulerMode) {
      const hit = rulerHitAtClient(ev.clientX, ev.clientY);
      canvas.style.cursor = hit ? (hit.kind === 'move' ? 'move' : 'grab') : 'crosshair';
    }
  });
  canvas.addEventListener('click', ev => {
    if (!rulerMode) return;
    ev.preventDefault();
    focusStage();
    if (rulerSuppressClick) {
      rulerSuppressClick = false;
      return;
    }
    if (mouseDragged) return;
    const p = clientToPreview(ev.clientX, ev.clientY);
    if (!ruler) {
      const point = previewToRulerPoint(p.px, p.py);
      ruler = { start: point, end: point, complete: false };
      updateRulerButtons();
      setRulerStatus();
      draw();
    } else if (!ruler.complete) {
      ruler.end = previewToRulerPoint(p.px, p.py, ruler.start && ruler.start.photoId);
      ruler.complete = true;
      updateRulerButtons();
      setRulerStatus();
      draw();
    }
  });
  window.addEventListener('mousemove', ev => {
    if (rulerDrag && ruler && ruler.complete) {
      hideTooltip();
      if (mouseDownPos && Math.hypot(ev.clientX - mouseDownPos[0], ev.clientY - mouseDownPos[1]) > 4) {
        mouseDragged = true;
        rulerSuppressClick = true;
      }
      const p = clientToPreview(ev.clientX, ev.clientY);
      const dx = p.px - rulerDrag.origin[0];
      const dy = p.py - rulerDrag.origin[1];
      if (rulerDrag.kind === 'start' || rulerDrag.kind === 'move') {
        ruler.start = previewToRulerPoint(
          rulerDrag.startPreview[0] + dx,
          rulerDrag.startPreview[1] + dy,
          rulerDrag.startPhotoId
        );
      }
      if (rulerDrag.kind === 'end' || rulerDrag.kind === 'move') {
        ruler.end = previewToRulerPoint(
          rulerDrag.endPreview[0] + dx,
          rulerDrag.endPreview[1] + dy,
          rulerDrag.endPhotoId
        );
      }
      setRulerStatus();
      draw();
      return;
    }
    if (!dragging) {
      if (rulerMode) {
        hideTooltip();
        const hit = rulerHitAtClient(ev.clientX, ev.clientY);
        canvas.style.cursor = hit ? (hit.kind === 'move' ? 'move' : 'grab') : 'crosshair';
        return;
      }
      if (!state || !state.photos) {
        hideTooltip();
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width;
      const sy = canvas.height / rect.height;
      const px = pan[0] + ((ev.clientX - rect.left) * sx) / zoom;
      const py = pan[1] + ((ev.clientY - rect.top) * sy) / zoom;
      let found = null;
      let under = null;
      for (let i = state.photos.length - 1; i >= 0; i -= 1) {
        const photo = state.photos[i];
        if (pointInPhoto(px, py, photo)) {
          if (!found) found = photo;
          else {
            under = photo;
            break;
          }
        }
      }
      const nextHoverId = found ? found.id : null;
      const nextUnderId = under ? under.id : null;
      hoverScreenX = (ev.clientX - rect.left) * sx;
      if (hoverPhotoId !== nextHoverId || hoverUnderPhotoId !== nextUnderId) {
        hoverPhotoId = nextHoverId;
        hoverUnderPhotoId = nextUnderId;
        const mode = compareMode();
        if (pairModeActive(mode)) draw();
      } else if (compareMode() === 'pair-swipe') {
        draw();
      }
      if (altitudeShiftArrowHitAtClient(ev.clientX, ev.clientY)) {
        showTooltip(ev, altitudeShiftArrowWarningText);
      } else if (found) {
        showPhotoTooltip(ev, found);
      } else {
        hideTooltip();
      }
      return;
    }
    hideTooltip();
    if (mouseDownPos && Math.hypot(ev.clientX - mouseDownPos[0], ev.clientY - mouseDownPos[1]) > 4) {
      mouseDragged = true;
    }
    if (hoverPhotoId !== null || hoverUnderPhotoId !== null) {
      hoverPhotoId = null;
      hoverUnderPhotoId = null;
      if (pairModeActive(compareMode())) draw();
    }
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    pan[0] -= (ev.clientX - lastMouse[0]) * sx / zoom;
    pan[1] -= (ev.clientY - lastMouse[1]) * sy / zoom;
    lastMouse = [ev.clientX, ev.clientY];
    draw();
  });
  window.addEventListener('resize', resize);
  window.addEventListener('pagehide', notifyServerShutdown);
  window.addEventListener('beforeunload', notifyServerShutdown);
  canvas.addEventListener('mouseleave', () => {
    hideTooltip();
    if (rulerMode && !rulerDrag) canvas.style.cursor = 'crosshair';
    if (hoverPhotoId !== null || hoverUnderPhotoId !== null) {
      hoverPhotoId = null;
      hoverUnderPhotoId = null;
      if (pairModeActive(compareMode())) draw();
    }
  });
  applyStaticI18n();
  updateRulerButtons();
  resize();
  setMemoryPolling(5000);
  updateMemoryInfo();
  loadState(true).catch(err => {
    console.error(err);
    hud.textContent = String(err);
  });
})();"""


def _render_options_from_args(args: argparse.Namespace) -> RenderOptions:
    hfov_deg = getattr(args, "hfov_deg", None)
    if hfov_deg is not None and not (0.0 < float(hfov_deg) < 180.0):
        raise SystemExit("--hfov-deg/--fov must be greater than 0 and less than 180")
    return RenderOptions(
        undistort=bool(getattr(args, "undistort", False)),
        k1=float(getattr(args, "k1", 0.1400)),
        k2=float(getattr(args, "k2", -0.3979)),
        k3=float(getattr(args, "k3", 0.4837)),
        alt_correction_m=float(getattr(args, "alt_correction", 0.0)),
        yaw_offset_deg=float(getattr(args, "yaw_offset", 0.0)),
        yaw_invert=bool(getattr(args, "yaw_invert", False)),
        yaw_both=bool(getattr(args, "yaw_both", False)),
        yaw_gimbal_only=bool(getattr(args, "yaw_gimbal_only", False)),
        yaw_flight_only=bool(getattr(args, "yaw_flight_only", False)),
        opacity_pct=float(getattr(args, "opacity", 100.0)),
        roi_warp=not bool(getattr(args, "no_roi_warp", False)),
        roi_margin_px=int(getattr(args, "roi_margin", 8)),
        jpg_quality=int(getattr(args, "jpg_quality", 95)),
        png_compress_level=int(getattr(args, "png_compress_level", 6)),
        preview_max_dim=int(getattr(args, "preview_max_pix", 2048)),
        use_pitch=bool(getattr(args, "use_pitch", False)),
        crop_optimize=bool(getattr(args, "crop_optimize", False)),
        hfov_deg_override=None if hfov_deg is None else float(hfov_deg),
    )


def _copy_options(
    options: RenderOptions,
    *,
    alt_correction_m: Optional[float] = None,
    yaw_offset_deg: Optional[float] = None,
    yaw_mode: Optional[str] = None,
    yaw_invert: Optional[bool] = None,
    opacity_pct: Optional[float] = None,
    crop_optimize: Optional[bool] = None,
) -> RenderOptions:
    mode = yaw_mode or _webui_yaw_mode(options)
    return RenderOptions(
        undistort=options.undistort,
        k1=options.k1,
        k2=options.k2,
        k3=options.k3,
        alt_correction_m=options.alt_correction_m if alt_correction_m is None else float(alt_correction_m),
        yaw_offset_deg=options.yaw_offset_deg if yaw_offset_deg is None else float(yaw_offset_deg),
        yaw_invert=options.yaw_invert if yaw_invert is None else bool(yaw_invert),
        yaw_both=(mode == "both"),
        yaw_gimbal_only=(mode == "gimbal_only"),
        yaw_flight_only=(mode == "flight_only"),
        opacity_pct=options.opacity_pct if opacity_pct is None else max(0.0, min(100.0, float(opacity_pct))),
        roi_warp=options.roi_warp,
        roi_margin_px=options.roi_margin_px,
        jpg_quality=options.jpg_quality,
        png_compress_level=options.png_compress_level,
        preview_max_dim=options.preview_max_dim,
        use_pitch=options.use_pitch,
        crop_optimize=options.crop_optimize if crop_optimize is None else bool(crop_optimize),
        hfov_deg_override=options.hfov_deg_override,
    )


def _webui_yaw_mode(options: RenderOptions) -> str:
    if bool(getattr(options, "yaw_gimbal_only", False)):
        return "gimbal_only"
    if bool(getattr(options, "yaw_both", False)):
        return "both"
    return "flight_only"


def _stats_line(values: List[Optional[float]]) -> str:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return "N/A"
    return f"avg={float(np.mean(nums)):.2f}, min={float(np.min(nums)):.2f}, max={float(np.max(nums)):.2f}"


def _first_text_or_na(values: List[Optional[str]]) -> str:
    for v in values:
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return "N/A"


def _abs_rel_alt_difference_m(meta: PhotoMeta) -> Optional[float]:
    absolute_alt = getattr(meta, "absolute_alt_m", None)
    if absolute_alt is None:
        return None
    return float(absolute_alt) - float(meta.alt_m)


def _altitude_shift_boundary_indices(ordered_sequence: List[Tuple[int, PhotoMeta]]) -> List[int]:
    boundary_indices: List[int] = []
    previous_difference: Optional[float] = None
    for sequence_index, (_, meta) in enumerate(ordered_sequence):
        difference = _abs_rel_alt_difference_m(meta)
        if difference is None:
            previous_difference = None
            continue
        if (
            previous_difference is not None
            and abs(difference - previous_difference) >= (ABS_REL_ALT_CHANGE_THRESHOLD_M - 1e-9)
        ):
            boundary_indices.append(sequence_index)
        previous_difference = difference
    return boundary_indices


def _build_ui_info(metas: List[PhotoMeta], options: RenderOptions) -> Dict[str, List[str]]:
    if not metas:
        return {
            "camera_water_lines": ["Avg. Camera-Water Distance: 0.00 m"],
            "relative_altitude_lines": [
                "Relative Avg: 0.00 m",
                "Relative Min: 0.00 m",
                "Relative Max: 0.00 m",
                "Absolute Avg: N/A",
                "Absolute Min: N/A",
                "Absolute Max: N/A",
            ],
            "yaw_description_lines": [
                "Flight Yaw Degree = Drone direction (Exif)",
                "Gimbal Yaw Degree = Gimbal direction (Exif)",
            ],
            "shortcut_lines": ["H: Altitude +0.2m", "J: Altitude -0.2m", "K: Rotation +2 deg", "L: Rotation -2 deg"],
            "exif_lines": ["No EXIF stats available"],
        }

    rel_vals = [float(m.alt_m) for m in metas]
    rel_avg = float(np.mean(rel_vals))
    rel_min = float(np.min(rel_vals))
    rel_max = float(np.max(rel_vals))
    cam_water_avg = rel_avg + float(getattr(options, "alt_correction_m", 0.0))
    abs_vals = [float(m.absolute_alt_m) for m in metas if getattr(m, "absolute_alt_m", None) is not None]
    if abs_vals:
        absolute_altitude_lines = [
            f"Absolute Avg: {float(np.mean(abs_vals)):.2f} m",
            f"Absolute Min: {float(np.min(abs_vals)):.2f} m",
            f"Absolute Max: {float(np.max(abs_vals)):.2f} m",
        ]
    else:
        absolute_altitude_lines = ["Absolute Avg: N/A", "Absolute Min: N/A", "Absolute Max: N/A"]

    fov_vals = [float(m.hfov_deg) for m in metas]
    fov_avg = float(np.mean(fov_vals))
    fov_min = float(np.min(fov_vals))
    fov_max = float(np.max(fov_vals))
    if (fov_max - fov_min) <= 1e-3:
        fov_line = f"FOV [deg]: {fov_avg:.2f} (uniform)"
    else:
        fov_line = f"FOV [deg]: mismatch detected (avg={fov_avg:.2f}, min={fov_min:.2f}, max={fov_max:.2f})"

    return {
        "camera_water_lines": [f"Avg. Camera-Water Distance: {cam_water_avg:.2f} m"],
        "relative_altitude_lines": [
            f"Relative Avg: {rel_avg:.2f} m",
            f"Relative Min: {rel_min:.2f} m",
            f"Relative Max: {rel_max:.2f} m",
        ] + absolute_altitude_lines,
        "yaw_description_lines": [
            "Flight Yaw Degree = Drone direction (Exif)",
            "Gimbal Yaw Degree = Gimbal direction (Exif)",
        ],
        "shortcut_lines": [
            "H: Altitude +0.2m",
            "J: Altitude -0.2m",
            "K: Rotation +2 deg",
            "L: Rotation -2 deg",
        ],
        "exif_lines": [
            "Product Name: " + _first_text_or_na([getattr(m, "product_name", None) for m in metas]),
            "Unique Camera Model: " + _first_text_or_na([getattr(m, "unique_camera_model", None) for m in metas]),
            fov_line,
            "Flight Pitch [deg]: " + _stats_line([getattr(m, "flight_pitch_deg", None) for m in metas]),
            "Gimbal Pitch [deg]: " + _stats_line([getattr(m, "gimbal_pitch_deg", None) for m in metas]),
            "Gimbal Roll [degree]: " + _stats_line([getattr(m, "gimbal_roll_deg", None) for m in metas]),
            "Flight Roll [degree]: " + _stats_line([getattr(m, "flight_roll_deg", None) for m in metas]),
            "Flight Yaw [deg]: " + _stats_line([getattr(m, "flight_yaw_deg", None) for m in metas]),
            "Gimbal Yaw [deg]: " + _stats_line([getattr(m, "gimbal_yaw_deg", None) for m in metas]),
            "Flight X Speed [m/s]: " + _stats_line([getattr(m, "flight_x_speed_mps", None) for m in metas]),
            "Flight Y Speed [m/s]: " + _stats_line([getattr(m, "flight_y_speed_mps", None) for m in metas]),
            "Flight Z Speed [m/s]: " + _stats_line([getattr(m, "flight_z_speed_mps", None) for m in metas]),
        ],
    }


def _build_area_stats(
    polygons_full_px: List[List[Tuple[float, float]]],
    canvas_w: int,
    canvas_h: int,
    mpp: float,
) -> Dict[str, object]:
    if not polygons_full_px or canvas_w <= 0 or canvas_h <= 0 or mpp <= 0:
        return {
            "mosaic_area_m2": 0.0,
            "overlap_area_m2": 0.0,
            "overlap_pct": 0.0,
            "area_error_m2": 0.0,
            "overlap_pct_error": 0.0,
            "approximate": False,
        }

    mask_scale = min(1.0, AREA_MASK_MAX_DIM_PX / float(max(canvas_w, canvas_h)))
    mask_w = max(1, int(math.ceil(canvas_w * mask_scale)))
    mask_h = max(1, int(math.ceil(canvas_h * mask_scale)))
    count_dtype = np.uint16 if len(polygons_full_px) <= np.iinfo(np.uint16).max else np.uint32
    coverage = np.zeros((mask_h, mask_w), dtype=count_dtype)
    mask = Image.new("1", (mask_w, mask_h), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons_full_px:
        mask.paste(0, (0, 0, mask_w, mask_h))
        draw.polygon([(x * mask_scale, y * mask_scale) for x, y in polygon], fill=1)
        coverage += np.asarray(mask, dtype=count_dtype)

    area_per_cell_m2 = (float(mpp) / mask_scale) ** 2
    mosaic_area_m2 = float(np.count_nonzero(coverage >= 1)) * area_per_cell_m2
    overlap_area_m2 = float(np.count_nonzero(coverage >= 2)) * area_per_cell_m2
    overlap_pct = 0.0 if mosaic_area_m2 <= 0 else (overlap_area_m2 / mosaic_area_m2) * 100.0
    cell_size_m = float(mpp) / mask_scale
    perimeter_px = 0.0
    for polygon in polygons_full_px:
        for i, p0 in enumerate(polygon):
            p1 = polygon[(i + 1) % len(polygon)]
            perimeter_px += math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
    area_error_m2 = perimeter_px * float(mpp) * cell_size_m * AREA_ERROR_SAFETY_FACTOR
    if mosaic_area_m2 > 0:
        overlap_pct_error = 100.0 * (
            (area_error_m2 / mosaic_area_m2)
            + ((overlap_area_m2 * area_error_m2) / (mosaic_area_m2 * mosaic_area_m2))
        )
        overlap_pct_error = min(100.0, overlap_pct_error)
    else:
        overlap_pct_error = 0.0
    return {
        "mosaic_area_m2": mosaic_area_m2,
        "overlap_area_m2": overlap_area_m2,
        "overlap_pct": overlap_pct,
        "area_error_m2": area_error_m2,
        "overlap_pct_error": overlap_pct_error,
        "approximate": mask_scale < 1.0,
    }


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _first_descendant(elem: ET.Element, local_name: str) -> Optional[ET.Element]:
    for child in elem.iter():
        if _xml_local_name(child.tag) == local_name:
            return child
    return None


def _first_descendant_text(elem: ET.Element, local_name: str) -> Optional[str]:
    child = _first_descendant(elem, local_name)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _float_text(elem: ET.Element, local_name: str) -> Optional[float]:
    text = _first_descendant_text(elem, local_name)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_kml_coordinates(text: str) -> List[Dict[str, float]]:
    coords: List[Dict[str, float]] = []
    for item in text.replace("\n", " ").replace("\t", " ").split():
        parts = item.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        coords.append({"lon": lon, "lat": lat})
    return coords


def _safe_kmz_member_path(base_name: str, href: str) -> str:
    href = href.strip().replace("\\", "/")
    if not href or "://" in href or href.startswith("/"):
        raise FileNotFoundError("external or absolute GroundOverlay icon href is not supported")
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(base_name), href))
    if joined.startswith("../") or joined == "..":
        raise FileNotFoundError("invalid GroundOverlay icon href")
    return joined


def _parse_kmz_preview_features(data: bytes, display_name: str) -> Dict[str, object]:
    items: List[Dict[str, object]] = []
    markers: List[Dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        kml_names = [n for n in names if n.lower().endswith(".kml")]
        if not kml_names:
            raise ValueError("KMZ does not contain a KML file")
        kml_name = "doc.kml" if "doc.kml" in kml_names else kml_names[0]
        root = ET.fromstring(zf.read(kml_name))
        for placemark in root.iter():
            if _xml_local_name(placemark.tag) != "Placemark":
                continue
            point = _first_descendant(placemark, "Point")
            if point is None:
                continue
            coord_text = _first_descendant_text(point, "coordinates")
            if not coord_text:
                continue
            coords = _parse_kml_coordinates(coord_text)
            if not coords:
                continue
            coord = coords[0]
            markers.append(
                {
                    "name": _first_descendant_text(placemark, "name") or "",
                    "lat": coord["lat"],
                    "lon": coord["lon"],
                }
            )
        for overlay in root.iter():
            if _xml_local_name(overlay.tag) != "GroundOverlay":
                continue
            icon = _first_descendant(overlay, "Icon")
            href = _first_descendant_text(icon, "href") if icon is not None else None
            if not href:
                continue
            member_name = _safe_kmz_member_path(kml_name, href)
            if member_name not in names:
                continue
            image_data = zf.read(member_name)
            mime = mimetypes.guess_type(member_name)[0] or "application/octet-stream"
            item: Dict[str, object] = {
                "name": _first_descendant_text(overlay, "name") or posixpath.basename(member_name),
                "data_url": f"data:{mime};base64,{base64.b64encode(image_data).decode('ascii')}",
            }
            quad = _first_descendant(overlay, "LatLonQuad")
            quad_coords = _first_descendant_text(quad, "coordinates") if quad is not None else None
            if quad_coords:
                coords = _parse_kml_coordinates(quad_coords)
                if len(coords) >= 4:
                    # gx:LatLonQuad is lower-left, lower-right, upper-right, upper-left.
                    # The canvas mapper expects upper-left, upper-right, lower-right, lower-left.
                    item["coordinates"] = [coords[3], coords[2], coords[1], coords[0]]
            if "coordinates" not in item:
                box = _first_descendant(overlay, "LatLonBox")
                if box is None:
                    continue
                north = _float_text(box, "north")
                south = _float_text(box, "south")
                east = _float_text(box, "east")
                west = _float_text(box, "west")
                if None in (north, south, east, west):
                    continue
                item["lat_lon_box"] = {
                    "north": north,
                    "south": south,
                    "east": east,
                    "west": west,
                    "rotation": _float_text(box, "rotation") or 0.0,
                }
            items.append(item)
    return {"ok": True, "name": display_name, "items": items, "markers": markers}


@dataclass
class WebSaveJob:
    total: int
    done: int = 0
    current: str = ""
    phase: str = "Preparing"
    status: str = "running"
    path: str = ""
    error: Optional[str] = None
    save_state: SaveProgressState = field(default_factory=SaveProgressState)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(
        self,
        *,
        phase: Optional[str] = None,
        done: Optional[int] = None,
        current: Optional[str] = None,
        status: Optional[str] = None,
        path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self.lock:
            if phase is not None:
                self.phase = phase
            if done is not None:
                self.done = int(done)
            if current is not None:
                self.current = current
            if status is not None:
                self.status = status
            if path is not None:
                self.path = path
            if error is not None:
                self.error = error

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise SaveCancelledError("Save cancelled")

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            payload = {
                "active": True,
                "status": self.status,
                "phase": self.phase,
                "done": self.done,
                "total": self.total,
                "current": self.current,
                "path": self.path,
                "error": self.error,
            }
        payload["save_text"] = _format_save_progress_text(self.save_state)
        return payload


class WebMosaicSession:
    def __init__(self, in_dir: str, out_path: str, options: RenderOptions):
        self.in_dir = in_dir
        self.out_path = out_path
        self.initial_options = options
        self.options = options
        self.lock = threading.Lock()
        self.save_job: Optional[WebSaveJob] = None
        self.image_paths: List[str] = []
        self.metas: List[PhotoMeta] = []
        self._reload_metas()

    def _reload_metas(self) -> None:
        paths = _list_images(self.in_dir)
        metas: List[PhotoMeta] = []
        image_paths: List[str] = []
        for p in paths:
            try:
                meta = _load_photo_meta(p, self.options)
            except Exception:
                meta = None
            if meta is None:
                continue
            image_paths.append(p)
            metas.append(meta)
        if not metas:
            raise SystemExit("No usable images found.")
        self.image_paths = image_paths
        self.metas = metas

    def update_options(self, payload: Dict[str, object]) -> Dict[str, object]:
        with self.lock:
            old_mode = _webui_yaw_mode(self.options)
            old_invert = bool(self.options.yaw_invert)
            self.options = _copy_options(
                self.options,
                alt_correction_m=float(payload.get("alt_correction_m", self.options.alt_correction_m)),
                yaw_offset_deg=float(payload.get("yaw_offset_deg", self.options.yaw_offset_deg)),
                yaw_mode=str(payload.get("yaw_mode", old_mode)),
                yaw_invert=bool(payload.get("yaw_invert", old_invert)),
                opacity_pct=float(payload.get("opacity_pct", self.options.opacity_pct)),
                crop_optimize=bool(payload.get("crop_optimize", self.options.crop_optimize)),
            )
            if old_mode != _webui_yaw_mode(self.options) or old_invert != bool(self.options.yaw_invert):
                self._reload_metas()
            return self.state()

    def revert_options(self) -> Dict[str, object]:
        with self.lock:
            self.options = self.initial_options
            self._reload_metas()
            return self.state()

    def state(self) -> Dict[str, object]:
        if not self.metas:
            self._reload_metas()
        origin_lat, origin_lon, min_x, min_y, max_x, max_y, mpp = _compute_canvas(self.metas, self.options)
        canvas_w = int(math.ceil((max_x - min_x) / mpp))
        canvas_h = int(math.ceil((max_y - min_y) / mpp))
        preview_max = max(64, int(getattr(self.options, "preview_max_dim", 2048)))
        scale = min(1.0, preview_max / max(canvas_w, canvas_h))
        use_crop_opt = bool(getattr(self.options, "crop_optimize", False))
        canvas_memory = _estimated_canvas_memory_bytes(canvas_w, canvas_h, self.out_path, use_crop_opt)
        peak_photo_memory = 0 if not self.metas else max(
            _estimate_photo_peak_bytes(meta, canvas_w, canvas_h, mpp, self.options) for meta in self.metas
        )
        estimated_memory = canvas_memory + peak_photo_memory
        m_per_deg_lat, m_per_deg_lon = _meters_per_deg(origin_lat)
        photos: List[Dict[str, object]] = []
        sequence: List[Dict[str, object]] = []
        ordered_sequence = sorted(
            enumerate(self.metas),
            key=lambda item: ((getattr(item[1], "captured_at", None) or ""), os.path.basename(item[1].path), item[0]),
        )
        altitude_shift_boundaries = _altitude_shift_boundary_indices(ordered_sequence)
        first_altitude_shift_boundary = altitude_shift_boundaries[0] if altitude_shift_boundaries else None
        altitude_shift_ids = (
            {idx for idx, _ in ordered_sequence[first_altitude_shift_boundary:]}
            if first_altitude_shift_boundary is not None
            else set()
        )
        ordered = sorted(enumerate(self.metas), key=lambda item: _effective_alt_m(item[1], self.options), reverse=True)
        for idx, meta in ordered:
            corners_world = _project_corners(meta, origin_lat, origin_lon, self.options)
            u, v = _world_to_canvas(
                corners_world[:, 0],
                corners_world[:, 1],
                min_x=min_x,
                max_y=max_y,
                mpp=mpp,
            )
            corners = [[float(u[i]) * scale, float(v[i]) * scale] for i in range(4)]
            photos.append(
                {
                    "id": idx,
                    "name": os.path.basename(meta.path),
                    "url": f"/image/{idx}",
                    "corners": corners,
                    "width": meta.w,
                    "height": meta.h,
                    "alt_m": meta.alt_m,
                    "absolute_alt_m": meta.absolute_alt_m,
                    "yaw_deg": meta.yaw_deg,
                    "flight_pitch_deg": meta.flight_pitch_deg,
                    "gimbal_pitch_deg": meta.gimbal_pitch_deg,
                    "flight_roll_deg": meta.flight_roll_deg,
                    "flight_yaw_deg": meta.flight_yaw_deg,
                    "gimbal_yaw_deg": meta.gimbal_yaw_deg,
                    "flight_x_speed_mps": meta.flight_x_speed_mps,
                    "flight_y_speed_mps": meta.flight_y_speed_mps,
                    "flight_z_speed_mps": meta.flight_z_speed_mps,
                    "altitude_shift_alert": idx in altitude_shift_ids,
                }
            )
        for sequence_index, (idx, meta) in enumerate(ordered_sequence):
            corners_world = _project_corners(meta, origin_lat, origin_lon, self.options)
            center_world = np.mean(corners_world, axis=0)
            center_u, center_v = _world_to_canvas(
                np.array([center_world[0]], dtype=np.float64),
                np.array([center_world[1]], dtype=np.float64),
                min_x=min_x,
                max_y=max_y,
                mpp=mpp,
            )
            sequence.append(
                {
                    "id": idx,
                    "order": sequence_index + 1,
                    "name": os.path.basename(meta.path),
                    "captured_at": getattr(meta, "captured_at", None) or os.path.basename(meta.path),
                    "center": [float(center_u[0]) * scale, float(center_v[0]) * scale],
                    "altitude_shift_from_previous": sequence_index in altitude_shift_boundaries,
                }
            )
        return {
            "canvas": {"width": canvas_w * scale, "height": canvas_h * scale, "scale": scale},
            "full_canvas": {
                "width": canvas_w,
                "height": canvas_h,
                "mpp": mpp,
                "estimated_memory_bytes": estimated_memory,
                "estimated_memory": _format_bytes_hr(estimated_memory),
                "memory_safety_margin_bytes": MEMORY_SAFETY_MARGIN_BYTES,
            },
            "georef": {
                "origin_lat": origin_lat,
                "origin_lon": origin_lon,
                "min_x": min_x,
                "max_y": max_y,
                "m_per_deg_lat": m_per_deg_lat,
                "m_per_deg_lon": m_per_deg_lon,
            },
            "options": {
                "alt_correction_m": self.options.alt_correction_m,
                "yaw_offset_deg": self.options.yaw_offset_deg,
                "yaw_mode": _webui_yaw_mode(self.options),
                "yaw_invert": self.options.yaw_invert,
                "opacity_pct": self.options.opacity_pct,
                "undistort": self.options.undistort,
                "crop_optimize": self.options.crop_optimize,
            },
            "info": _build_ui_info(self.metas, self.options),
            "photos": photos,
            "sequence": sequence,
            "warnings": ["WebUI preview does not apply undistortion yet."] if self.options.undistort else [],
        }

    def area_stats(self) -> Dict[str, object]:
        with self.lock:
            if not self.metas:
                self._reload_metas()
            metas = list(self.metas)
            options = self.options
        origin_lat, origin_lon, min_x, min_y, max_x, max_y, mpp = _compute_canvas(metas, options)
        canvas_w = int(math.ceil((max_x - min_x) / mpp))
        canvas_h = int(math.ceil((max_y - min_y) / mpp))
        polygons_full_px: List[List[Tuple[float, float]]] = []
        for meta in metas:
            corners_world = _project_corners(meta, origin_lat, origin_lon, options)
            u, v = _world_to_canvas(
                corners_world[:, 0],
                corners_world[:, 1],
                min_x=min_x,
                max_y=max_y,
                mpp=mpp,
            )
            polygons_full_px.append([(float(u[i]), float(v[i])) for i in range(4)])
        return _build_area_stats(polygons_full_px, canvas_w, canvas_h, mpp)

    def image_response(self, image_id: int) -> Tuple[bytes, str]:
        if image_id < 0 or image_id >= len(self.image_paths):
            raise FileNotFoundError(f"image id not found: {image_id}")
        path = self.image_paths[image_id]
        with open(path, "rb") as fh:
            data = fh.read()
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return data, mime

    def save(self, force: bool = False) -> Dict[str, object]:
        with self.lock:
            if self.save_job is not None and self.save_job.snapshot().get("status") in ("running", "saving"):
                return self.save_job.snapshot()
            save_opts = self.options
            metas = list(self.metas)
            job = WebSaveJob(total=max(1, len(metas)))
            self.save_job = job
        thread = threading.Thread(target=self._run_save_job, args=(job, metas, save_opts, force), daemon=True)
        thread.start()
        return job.snapshot()

    def save_status(self) -> Dict[str, object]:
        job = self.save_job
        if job is None:
            return {"active": False}
        return job.snapshot()

    def memory_status(self) -> Dict[str, object]:
        total = _system_mem_total_bytes()
        available = _memory_headroom_bytes()
        pct = 0.0
        if total is not None and total > 0 and available is not None:
            pct = max(0.0, min(100.0, (float(available) / float(total)) * 100.0))
        return {
            "total_bytes": total,
            "available_bytes": available,
            "total": "N/A" if total is None else _format_bytes_hr(total),
            "available": "N/A" if available is None else _format_bytes_hr(available),
            "available_pct": pct,
            "is_wsl": _is_wsl(),
        }

    def cancel_save(self) -> Dict[str, object]:
        job = self.save_job
        if job is None:
            return {"active": False}
        job.cancel_event.set()
        return job.snapshot()

    def _run_save_job(
        self,
        job: WebSaveJob,
        metas: List[PhotoMeta],
        save_opts: RenderOptions,
        force: bool = False,
    ) -> None:
        try:
            job.check_cancelled()
            job.update(phase="Computing canvas", done=0, current="")
            origin_lat, origin_lon, min_x, min_y, max_x, max_y, mpp = _compute_canvas(metas, save_opts)
            canvas_w = int(math.ceil((max_x - min_x) / mpp))
            canvas_h = int(math.ceil((max_y - min_y) / mpp))
            _ensure_output_dimension_limit(self.out_path, canvas_w, canvas_h)
            if not force:
                _ensure_save_memory_headroom(canvas_w, canvas_h, self.out_path, metas, save_opts, mpp)

            job.check_cancelled()
            base = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            metas_sorted = sorted(metas, key=lambda m: _effective_alt_m(m, save_opts), reverse=True)
            job.total = max(1, len(metas_sorted))
            use_crop_opt = bool(getattr(save_opts, "crop_optimize", False))
            best_dist2: Optional[np.ndarray] = None
            if use_crop_opt:
                best_dist2 = np.full((canvas_h, canvas_w), np.inf, dtype=np.float32)

            for i, meta in enumerate(metas_sorted):
                job.check_cancelled()
                fname = os.path.basename(meta.path)
                job.update(phase="Processing", done=i, current=fname)
                if not force:
                    _ensure_memory_headroom(
                        _estimate_photo_peak_bytes(meta, canvas_w, canvas_h, mpp, save_opts),
                        context=f"processing {fname}",
                        margin_bytes=MEMORY_SAFETY_MARGIN_BYTES // 2,
                    )
                warped, offset = _warp_photo_to_canvas(
                    meta,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    min_x=min_x,
                    max_y=max_y,
                    mpp_out=mpp,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                    options=save_opts,
                )
                job.check_cancelled()
                if use_crop_opt and best_dist2 is not None:
                    cu, cv = _photo_center_canvas_xy(
                        meta,
                        origin_lat=origin_lat,
                        origin_lon=origin_lon,
                        min_x=min_x,
                        max_y=max_y,
                        mpp_out=mpp,
                        options=save_opts,
                    )
                    _composite_nearest_center_rgba_inplace(base, best_dist2, warped, offset, cu, cv)
                else:
                    _alpha_blend_rgba_over_rgba_inplace(base, warped, offset)
                del warped
                job.update(done=i + 1, current=fname)

            job.check_cancelled()
            job.update(phase="Saving", status="saving", done=job.total, current="")
            _save_image_job(base, self.out_path, save_opts, job.save_state)
            job.update(status="done", phase="Done", path=os.path.abspath(self.out_path), current="")
        except SaveCancelledError as exc:
            job.update(status="cancelled", phase="Cancelled", error=str(exc))
        except BaseException as exc:
            job.update(status="error", phase="Error", error=str(exc))
        finally:
            gc.collect()


def _make_webui_handler(session: WebMosaicSession, debug: bool = False):
    class WebUIHandler(BaseHTTPRequestHandler):
        server_version = "bvpp-webui/0.1"

        def log_message(self, fmt: str, *args) -> None:
            if debug:
                sys.stderr.write("[webui] " + (fmt % args) + "\n")

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _tile(self, provider: str, z: int, x: int, y: int) -> Tuple[bytes, str]:
            if provider == "osm":
                max_z = 19
                url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                mime = "image/png"
            elif provider == "gsi":
                max_z = 18
                url = f"https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
                mime = "image/jpeg"
            else:
                raise FileNotFoundError("unknown tile provider")
            if z < 0 or z > max_z or x < 0 or y < 0 or x >= 2**z or y >= 2**z:
                raise FileNotFoundError("tile out of range")
            req = Request(
                url,
                headers={
                    "User-Agent": "bvpp-webui/0.1 (+local preview)",
                    "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*",
                },
            )
            with urlopen(req, timeout=10) as resp:
                return resp.read(), mime

        def _json(self, payload: Dict[str, object], status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, WEBUI_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._send(200, WEBUI_JS.encode("utf-8"), "application/javascript; charset=utf-8")
                return
            if parsed.path == "/api/state":
                with session.lock:
                    self._json(session.state())
                return
            if parsed.path == "/api/area-stats":
                self._json(session.area_stats())
                return
            if parsed.path == "/api/save-status":
                self._json(session.save_status())
                return
            if parsed.path == "/api/memory":
                self._json(session.memory_status())
                return
            if parsed.path.startswith("/tile/"):
                try:
                    _, _, provider, zs, xs, ys = parsed.path.split("/", 5)
                    data, mime = self._tile(provider, int(zs), int(xs), int(ys))
                    self._send(200, data, mime)
                except Exception as exc:
                    self._json({"ok": False, "error": str(exc)}, 404)
                return
            if parsed.path.startswith("/image/"):
                try:
                    image_id = int(parsed.path.rsplit("/", 1)[-1])
                    data, mime = session.image_response(image_id)
                    self._send(200, data, mime)
                except Exception as exc:
                    self._json({"ok": False, "error": str(exc)}, 404)
                return
            self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            if parsed.path == "/api/kmz-overlay":
                try:
                    name = parse_qs(parsed.query).get("name", ["overlay.kmz"])[0]
                    self._json(_parse_kmz_preview_features(raw, os.path.basename(name) or "overlay.kmz"))
                except Exception as exc:
                    self._json({"ok": False, "error": str(exc)}, 400)
                return
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                payload = {}
            if parsed.path == "/api/options":
                try:
                    self._json(session.update_options(payload))
                except Exception as exc:
                    self._json({"ok": False, "error": str(exc)}, 400)
                return
            if parsed.path == "/api/revert":
                try:
                    self._json(session.revert_options())
                except Exception as exc:
                    self._json({"ok": False, "error": str(exc)}, 400)
                return
            if parsed.path == "/api/save":
                try:
                    self._json(session.save(force=bool(payload.get("force", False))))
                except Exception as exc:
                    self._json({"ok": False, "error": str(exc)}, 500)
                return
            if parsed.path == "/api/save-status":
                self._json(session.save_status())
                return
            if parsed.path == "/api/cancel-save":
                self._json(session.cancel_save())
                return
            if parsed.path == "/api/shutdown":
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._json({"ok": False, "error": "not found"}, 404)

    return WebUIHandler


def _is_wsl() -> bool:
    if "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ:
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as fh:
            version = fh.read().lower()
        return "microsoft" in version or "wsl" in version
    except Exception:
        return False


def _open_webui_url(url: str) -> None:
    if _is_wsl():
        for cmd_exe in ("cmd.exe", "/mnt/c/Windows/System32/cmd.exe"):
            try:
                subprocess.Popen(
                    [cmd_exe, "/C", "start", "", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                pass
    try:
        webbrowser.open(url)
    except Exception:
        pass


def launch_webui(
    in_dir: str,
    out_path: str,
    options: RenderOptions,
    host: str = "127.0.0.1",
    port: int = 8765,
    debug: bool = False,
) -> None:
    session = WebMosaicSession(in_dir, out_path, options)
    last_error: Optional[BaseException] = None
    httpd: Optional[ThreadingHTTPServer] = None
    for candidate in range(int(port), int(port) + 20):
        try:
            httpd = ThreadingHTTPServer((host, candidate), _make_webui_handler(session, debug=debug))
            port = candidate
            break
        except OSError as exc:
            last_error = exc
    if httpd is None:
        raise SystemExit(f"Failed to start WebUI server: {last_error}")

    url = f"http://{host}:{port}/"
    sys.stderr.write(f"bvpp WebUI ready: {url}\n")
    _open_webui_url(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("Stopping WebUI.\n")
    finally:
        httpd.server_close()


# ============================================================================
# GUI Implementation
# ============================================================================

# --- Preview-only warp helper (uses already decoded low-res RGBA; avoids Image.open on every preview update) ---
def _warp_photo_to_canvas_from_rgba(
    src_rgba: Image.Image,
    meta: PhotoMeta,
    origin_lat: float,
    origin_lon: float,
    min_x: float,
    max_y: float,
    mpp_out: float,
    canvas_w: int,
    canvas_h: int,
    options: RenderOptions,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Warp one (already decoded) RGBA image into the global canvas (preview path)."""
    # (Copied from _warp_photo_to_canvas but without Image.open)
    m_per_deg_lat, m_per_deg_lon = _meters_per_deg(origin_lat)
    x0 = (meta.lon_deg - origin_lon) * m_per_deg_lon
    y0 = (meta.lat_deg - origin_lat) * m_per_deg_lat

    # Tilt center shift
    # Apply this shift only when use_pitch is set
    if bool(getattr(options, "use_pitch", False)):
        alt_eff = float(meta.alt_m) + float(getattr(options, "alt_correction_m", 0.0))
        tilt_deg = _tilt_from_nadir_deg(getattr(meta, "pitch_deg", -90.0))
        forward = alt_eff * math.tan(math.radians(tilt_deg))
        top_world = _yaw_to_top_world(meta.yaw_deg, options)
        x0 += float(top_world[0]) * forward
        y0 += float(top_world[1]) * forward

    mpp_in = _compute_mpp(meta, options)
    right_world, down_world = _yaw_to_image_axes(meta.yaw_deg, options)

    cx = meta.w / 2.0
    cy = meta.h / 2.0

    def src_to_world_xy(xp: float, yp: float) -> Tuple[float, float]:
        dx = (xp - cx) * mpp_in
        dy = (yp - cy) * mpp_in
        wpt = np.array([x0, y0], dtype=np.float64) + dx * right_world + dy * down_world
        return float(wpt[0]), float(wpt[1])

    use_roi = bool(getattr(options, "roi_warp", True))
    margin = int(getattr(options, "roi_margin_px", 8))

    corners_src = [
        (0.0, 0.0),
        (meta.w - 1.0, 0.0),
        (meta.w - 1.0, meta.h - 1.0),
        (0.0, meta.h - 1.0),
    ]

    uv = []
    for xp, yp in corners_src:
        wx, wy = src_to_world_xy(xp, yp)
        u, v = _world_to_canvas(
            np.array([wx], dtype=np.float64),
            np.array([wy], dtype=np.float64),
            min_x=min_x,
            max_y=max_y,
            mpp=mpp_out,
        )
        uv.append((float(u[0]), float(v[0])))

    umin = min(p[0] for p in uv)
    umax = max(p[0] for p in uv)
    vmin = min(p[1] for p in uv)
    vmax = max(p[1] for p in uv)

    if use_roi:
        x1 = max(0, int(math.floor(umin)) - margin)
        y1 = max(0, int(math.floor(vmin)) - margin)
        x2 = min(canvas_w, int(math.ceil(umax)) + margin)
        y2 = min(canvas_h, int(math.ceil(vmax)) + margin)
        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)
        out_size = (roi_w, roi_h)
        offset = (x1, y1)
    else:
        out_size = (canvas_w, canvas_h)
        offset = (0, 0)

    src3 = np.array([[0.0, 0.0], [meta.w - 1.0, 0.0], [0.0, meta.h - 1.0]], dtype=np.float64)

    dst3 = []
    for i in range(3):
        wx, wy = src_to_world_xy(float(src3[i, 0]), float(src3[i, 1]))
        u, v = _world_to_canvas(
            np.array([wx], dtype=np.float64),
            np.array([wy], dtype=np.float64),
            min_x=min_x,
            max_y=max_y,
            mpp=mpp_out,
        )
        du = float(u[0]) - float(offset[0])
        dv = float(v[0]) - float(offset[1])
        dst3.append([du, dv])
    dst3 = np.array(dst3, dtype=np.float64)

    def affine_from_points(p_src: np.ndarray, p_dst: np.ndarray) -> np.ndarray:
        x = p_src[:, 0]
        y = p_src[:, 1]
        u = p_dst[:, 0]
        v = p_dst[:, 1]
        A = np.array(
            [
                [x[0], y[0], 1, 0, 0, 0],
                [0, 0, 0, x[0], y[0], 1],
                [x[1], y[1], 1, 0, 0, 0],
                [0, 0, 0, x[1], y[1], 1],
                [x[2], y[2], 1, 0, 0, 0],
                [0, 0, 0, x[2], y[2], 1],
            ],
            dtype=np.float64,
        )
        b = np.array([u[0], v[0], u[1], v[1], u[2], v[2]], dtype=np.float64)
        params = np.linalg.solve(A, b)
        return np.array(
            [[params[0], params[1], params[2]], [params[3], params[4], params[5]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    M_src_to_dst = affine_from_points(src3, dst3)
    M_dst_to_src = np.linalg.inv(M_src_to_dst)
    a, b, c = float(M_dst_to_src[0, 0]), float(M_dst_to_src[0, 1]), float(M_dst_to_src[0, 2])
    d, e, f = float(M_dst_to_src[1, 0]), float(M_dst_to_src[1, 1]), float(M_dst_to_src[1, 2])

    warped = src_rgba.transform(
        out_size,
        Image.AFFINE,
        data=(a, b, c, d, e, f),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0, 0),
    )

    warped = _apply_opacity_rgba(warped, float(getattr(options, "opacity_pct", 100.0)))
    return warped, offset


class MosaicViewer(QtWidgets.QGraphicsView):
    """Graphics view for displaying and interacting with the mosaic image."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        
        self._pan_active = False
        self._pan_start = QtCore.QPoint()
        self._layer_regions = []  # List of (bbox, overlay_text) for hover detection
    
    def wheelEvent(self, event: QtGui.QWheelEvent):
        """Handle mouse wheel zoom."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
    
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pan_active = True
            self._pan_start = event.pos()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._pan_active:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            # Check if mouse is over any image layer
            scene_pos = self.mapToScene(event.pos())
            x, y = scene_pos.x(), scene_pos.y()
            
            # Find topmost layer at this position
            found = None
            for bbox, overlay_text in reversed(self._layer_regions):  # Check from top to bottom
                x0, y0, x1, y1 = bbox
                if x0 <= x <= x1 and y0 <= y <= y1:
                    found = overlay_text
                    break
            
            if found:
                QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), found, self)
            else:
                QtWidgets.QToolTip.hideText()
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pan_active = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)


class MosaicGUI(QtWidgets.QMainWindow):
    """Main GUI window for interactive mosaic adjustment."""
    
    def __init__(self, in_dir: str, out_path: str, options: RenderOptions):
        super().__init__()
        
        if not QT_AVAILABLE:
            raise RuntimeError("PyQt6 not available. Install with: pip install PyQt6")
        
        self.in_dir = in_dir
        self.out_path = out_path
        self.options = options
        
        # Transform state
        self.scale_factor = 1.0
        self.rotation_deg = 0.0
        self.preview_opacity = 100.0
        
        # Yaw settings (mutable copy for GUI adjustments)
        self.yaw_mode = "flight_only"  # "both", "flight_only", "gimbal_only"
        self.yaw_invert = bool(getattr(options, "yaw_invert", False))
        
        # Set initial yaw mode from options
        if bool(getattr(options, "yaw_gimbal_only", False)):
            self.yaw_mode = "gimbal_only"
        elif bool(getattr(options, "yaw_both", False)):
            self.yaw_mode = "both"
        else:
            # Default behavior is flight-only.
            self.yaw_mode = "flight_only"
        
        # GUI-editable values (CLI-equivalent)
        self.alt_correction_m_gui = float(getattr(options, "alt_correction_m", 0.0))
        self.yaw_offset_deg_gui = float(getattr(options, "yaw_offset_deg", 0.0))

        # Preview source cache: decoded+downscaled RGBA to avoid re-reading originals on every preview update
        self._preview_src_rgba: Dict[str, Image.Image] = {}

        # Load metadata
        sys.stderr.write("Loading image metadata...\n")
        self.metas = self._load_metas()
        
        if not self.metas:
            raise SystemExit("No usable images found.")
        
        # Compute canvas geometry
        self.origin_lat, self.origin_lon, self.min_x, self.min_y, self.max_x, self.max_y, self.mpp = _compute_canvas(
            self.metas, self.options
        )
        self.canvas_w = int(math.ceil((self.max_x - self.min_x) / self.mpp))
        self.canvas_h = int(math.ceil((self.max_y - self.min_y) / self.mpp))
        _ensure_output_dimension_limit(self.out_path, self.canvas_w, self.canvas_h)
        
        sys.stderr.write(f"Canvas size: {self.canvas_w} x {self.canvas_h}\n")
        
        # Precompute preview scale and geometry (used to keep preview fast)
        self._preview_max_dim = max(64, int(getattr(self.options, "preview_max_dim", 2048)))
        self._preview_scale = min(1.0, self._preview_max_dim / max(self.canvas_w, self.canvas_h))
        self._preview_w = int(self.canvas_w * self._preview_scale)
        self._preview_h = int(self.canvas_h * self._preview_scale)
        self._preview_mpp = self.mpp / self._preview_scale

        # Build preview source cache once (uses current preview scale)
        self._build_preview_source_cache()

        # Generate initial low-res preview (warped layers built from cached sources)
        sys.stderr.write("Generating preview...\n")
        self.full_res_image = None
        self.preview_image = self._generate_preview()
        
        self._init_ui()
        
        sys.stderr.write("GUI ready.\n")
    
    def _build_preview_source_cache(self) -> None:
        sys.stderr.write("Caching low-res sources...\n")
        s = float(getattr(self, "_preview_scale", 1.0))
        for meta in self.metas:
            p = meta.path
            if p in self._preview_src_rgba:
                continue
            try:
                with Image.open(p) as im:
                    im = im.convert("RGBA")
                    if self.options.undistort:
                        im = _undistort_rgba(im, self.options.k1, self.options.k2, self.options.k3)
                    if s < 1.0:
                        nw = max(1, int(round(im.width * s)))
                        nh = max(1, int(round(im.height * s)))
                        im = im.resize((nw, nh), resample=Image.Resampling.BILINEAR)
                    self._preview_src_rgba[p] = im.copy()
            except Exception:
                continue
        sys.stderr.write("Caching low-res sources... done\n")

    def _load_metas(self) -> List[PhotoMeta]:
        """Load photo metadata with current yaw settings."""
        paths = _list_images(self.in_dir)
        metas = []
        # Build current options with GUI yaw settings
        current_opts = self._get_current_options()
        for p in paths:
            try:
                m = _load_photo_meta(p, current_opts)
                if m:
                    metas.append(m)
            except Exception:
                pass
        return metas
    
    def _get_current_options(self) -> RenderOptions:
        """Build RenderOptions reflecting current GUI state."""
        return RenderOptions(
            undistort=self.options.undistort,
            k1=self.options.k1,
            k2=self.options.k2,
            k3=self.options.k3,
            alt_correction_m=float(self.alt_correction_m_gui),
            yaw_offset_deg=float(self.yaw_offset_deg_gui),
            yaw_invert=self.yaw_invert,
            yaw_both=(self.yaw_mode == "both"),
            yaw_gimbal_only=(self.yaw_mode == "gimbal_only"),
            yaw_flight_only=(self.yaw_mode == "flight_only"),
            opacity_pct=self.options.opacity_pct,
            roi_warp=self.options.roi_warp,
            roi_margin_px=self.options.roi_margin_px,
            jpg_quality=self.options.jpg_quality,
            png_compress_level=self.options.png_compress_level,
            preview_max_dim=int(getattr(self.options, "preview_max_dim", 2048)),
            use_pitch=bool(getattr(self.options, "use_pitch", False)),
            crop_optimize=bool(getattr(self.options, "crop_optimize", False)),
            hfov_deg_override=getattr(self.options, "hfov_deg_override", None),
        )
    
    def _generate_preview(self, max_dim: int = 2048) -> Image.Image:
        """Generate low-resolution preview without reading originals again."""
        preview_w = getattr(self, "_preview_w", None) or int(self.canvas_w)
        preview_h = getattr(self, "_preview_h", None) or int(self.canvas_h)
        preview_mpp = getattr(self, "_preview_mpp", None) or float(self.mpp)

        base = Image.new("RGB", (preview_w, preview_h), (255, 255, 255))
        current_opts = self._get_current_options()
        metas_sorted = sorted(self.metas, key=lambda m: _effective_alt_m(m, current_opts), reverse=True)

        self._preview_layers = []
        self._preview_layer_metas = []

        for meta in metas_sorted:
            src = self._preview_src_rgba.get(meta.path)
            if src is None:
                continue

            pm = PhotoMeta(
                path=meta.path,
                w=src.width,
                h=src.height,
                lat_deg=meta.lat_deg,
                lon_deg=meta.lon_deg,
                alt_m=meta.alt_m,
                absolute_alt_m=meta.absolute_alt_m,
                yaw_deg=meta.yaw_deg,
                hfov_deg=meta.hfov_deg,
                pitch_deg=meta.pitch_deg,
                flight_yaw_deg=meta.flight_yaw_deg,
                gimbal_yaw_deg=meta.gimbal_yaw_deg,
                flight_pitch_deg=meta.flight_pitch_deg,
                gimbal_pitch_deg=meta.gimbal_pitch_deg,
                gimbal_roll_deg=meta.gimbal_roll_deg,
                flight_roll_deg=meta.flight_roll_deg,
                flight_x_speed_mps=meta.flight_x_speed_mps,
                flight_y_speed_mps=meta.flight_y_speed_mps,
                flight_z_speed_mps=meta.flight_z_speed_mps,
                product_name=meta.product_name,
                unique_camera_model=meta.unique_camera_model,
                captured_at=meta.captured_at,
            )

            warped, offset = _warp_photo_to_canvas_from_rgba(
                src,
                pm,
                origin_lat=self.origin_lat,
                origin_lon=self.origin_lon,
                min_x=self.min_x,
                max_y=self.max_y,
                mpp_out=preview_mpp,
                canvas_w=preview_w,
                canvas_h=preview_h,
                options=current_opts,
            )

            self._preview_layers.append((warped, offset))
            self._preview_layer_metas.append(meta)
            _alpha_blend_rgba_over_rgb_inplace(base, warped, offset)

        return base

    def _release_preview_memory(self) -> None:
        """Release preview caches and Qt display objects before full-res save."""
        self.preview_image = None
        self._preview_layers = []
        self._preview_layer_metas = []
        self._preview_src_rgba.clear()
        if hasattr(self, "scene"):
            self.scene.clear()
        if hasattr(self, "view"):
            self.view._layer_regions = []
            self.view.viewport().update()
        gc.collect()

    def _show_preview_placeholder(self, message: str) -> None:
        """Show a temporary message while preview data is unavailable."""
        if not hasattr(self, "scene"):
            return
        self.scene.clear()
        text_item = self.scene.addText(message)
        font = text_item.font()
        font.setPointSize(16)
        text_item.setFont(font)
        text_item.setDefaultTextColor(QtGui.QColor("#555555"))
        rect = text_item.boundingRect()
        pad_x = 40.0
        pad_y = 30.0
        self.scene.setSceneRect(0, 0, rect.width() + pad_x * 2, rect.height() + pad_y * 2)
        text_item.setPos(pad_x, pad_y)
        if hasattr(self, "view"):
            self.view._layer_regions = []
            self.view.viewport().update()

    def _restore_preview_after_save(self) -> None:
        """Rebuild preview caches after a save attempt finishes."""
        sys.stderr.write("Restoring preview...\n")
        self._update_preview_geometry_for_current_options()
        self._build_preview_source_cache()
        self.preview_image = self._generate_preview()
        self._update_display()
        gc.collect()

    def _update_display(self):
        """Display current preview_image (already mosaicked) and update hover regions."""
        self.scene.clear()
        if self.preview_image is None:
            return

        regions = []
        for i, (warped, offset) in enumerate(self._preview_layers):
            meta = self._preview_layer_metas[i] if i < len(self._preview_layer_metas) else None
            if meta is None:
                continue
            bbox = (offset[0], offset[1], offset[0] + warped.width, offset[1] + warped.height)
            fp = (
                f"{float(meta.flight_pitch_deg):.2f}"
                if getattr(meta, "flight_pitch_deg", None) is not None
                else "N/A"
            )
            gp = (
                f"{float(meta.gimbal_pitch_deg):.2f}"
                if getattr(meta, "gimbal_pitch_deg", None) is not None
                else "N/A"
            )
            fy = (
                f"{float(meta.flight_yaw_deg):.2f}"
                if getattr(meta, "flight_yaw_deg", None) is not None
                else "N/A"
            )
            gy = (
                f"{float(meta.gimbal_yaw_deg):.2f}"
                if getattr(meta, "gimbal_yaw_deg", None) is not None
                else "N/A"
            )
            overlay_text = (
                f"{os.path.basename(meta.path)}\n"
                f"Relative Altitude [m]: {float(meta.alt_m):.2f}\n"
                f"Flight Pitch [degree]: {fp}\n"
                f"Gimbal Pitch [degree]: {gp}\n"
                f"Flight Yaw Degree: {fy}\n"
                f"Gimbal Yaw Degree: {gy}"
            )
            regions.append((bbox, overlay_text))
        self.view._layer_regions = regions

        img = self.preview_image
        img_bytes = img.tobytes("raw", "RGB")
        qimg = QtGui.QImage(img_bytes, img.width, img.height, img.width * 3, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        if not hasattr(self, "_first_display_done"):
            # First time: ensure it fits the window (previous code fitInView here is unreliable
            # because controls may not be laid out yet)
            self._first_display_done = True

    def _create_controls(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(panel)

        # Altitude correction input
        lake_group = QtWidgets.QGroupBox("Camera-Water Distance")
        lake_v = QtWidgets.QVBoxLayout()
        lake_h = QtWidgets.QHBoxLayout()
        self.alt_correction_input = QtWidgets.QLineEdit(f"{self.alt_correction_m_gui:.2f}")
        self.alt_correction_input.setMaximumWidth(90)
        self.alt_correction_input.returnPressed.connect(self._on_alt_correction_input)
        lake_h.addWidget(QtWidgets.QLabel("Altitude Correction [m]:"))
        lake_h.addWidget(self.alt_correction_input)
        lake_v.addLayout(lake_h)

        self.cam_water_avg_label = QtWidgets.QLabel("Avg. Camera-Water Distance: 0.00 m")
        rel_stats_group = QtWidgets.QGroupBox("Relative Altitude (Exif) stats")
        rel_stats_v = QtWidgets.QVBoxLayout()
        self.rel_alt_avg_label = QtWidgets.QLabel("Avg: 0.00 m")
        self.rel_alt_min_label = QtWidgets.QLabel("Min: 0.00 m")
        self.rel_alt_max_label = QtWidgets.QLabel("Max: 0.00 m")
        for lbl in (
            self.rel_alt_avg_label,
            self.rel_alt_min_label,
            self.rel_alt_max_label,
        ):
            lbl.setStyleSheet("font-size: 9pt; color: #666;")
            rel_stats_v.addWidget(lbl)
        rel_stats_group.setLayout(rel_stats_v)
        lake_v.addWidget(self.cam_water_avg_label)
        self.cam_water_avg_label.setStyleSheet("font-size: 9pt; color: #666;")
        lake_v.addWidget(rel_stats_group)

        lake_group.setLayout(lake_v)
        layout.addWidget(lake_group)

        # Yaw offset input
        yawoff_group = QtWidgets.QGroupBox("Rotation Correction")
        yawoff_layout = QtWidgets.QHBoxLayout()
        self.yaw_offset_input = QtWidgets.QLineEdit(f"{self.yaw_offset_deg_gui:.2f}")
        self.yaw_offset_input.setMaximumWidth(90)
        self.yaw_offset_input.returnPressed.connect(self._on_yaw_offset_input)
        yawoff_layout.addWidget(QtWidgets.QLabel("Degree:"))
        yawoff_layout.addWidget(self.yaw_offset_input)
        yawoff_group.setLayout(yawoff_layout)

        # Opacity control
        opacity_group = QtWidgets.QGroupBox("Transparency (%)")
        opacity_layout = QtWidgets.QHBoxLayout()
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.opacity_label = QtWidgets.QLabel("100%")
        self.opacity_label.setMinimumWidth(40)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        opacity_group.setLayout(opacity_layout)

        # Stack Rotation Correction and Transparency vertically.
        # Keep full content visible by using natural minimum heights.
        yawoff_group.setMinimumHeight(yawoff_group.sizeHint().height())
        opacity_group.setMinimumHeight(opacity_group.sizeHint().height())
        rot_opacity_stack = QtWidgets.QWidget()
        rot_opacity_layout = QtWidgets.QVBoxLayout(rot_opacity_stack)
        rot_opacity_layout.setContentsMargins(0, 0, 0, 0)
        rot_opacity_layout.setSpacing(6)
        rot_opacity_layout.addWidget(yawoff_group)
        rot_opacity_layout.addWidget(opacity_group)
        layout.addWidget(rot_opacity_stack)

        # Yaw mode control
        yaw_group = QtWidgets.QGroupBox("Image Rotation Rule")
        yaw_layout = QtWidgets.QVBoxLayout()

        self.yaw_mode_combo = QtWidgets.QComboBox()
        self.yaw_mode_combo.addItems([
            "Use Flight + Gimbal Yaw Degree",
            "Use Flight Yaw Degree",
            "Use Gimbal Yaw Degree",
        ])
        # Set initial selection from current yaw_mode
        if self.yaw_mode == "flight_only":
            self.yaw_mode_combo.setCurrentIndex(1)
        elif self.yaw_mode == "gimbal_only":
            self.yaw_mode_combo.setCurrentIndex(2)
        else:
            self.yaw_mode_combo.setCurrentIndex(0)
        self.yaw_mode_combo.currentIndexChanged.connect(self._on_yaw_mode_changed)

        # ADD: show dropdown in the layout
        yaw_layout.addWidget(self.yaw_mode_combo)

        self.yaw_invert_check = QtWidgets.QCheckBox("Reverse rotate")
        self.yaw_invert_check.setChecked(self.yaw_invert)
        self.yaw_invert_check.stateChanged.connect(self._on_yaw_invert_changed)
        yaw_layout.addWidget(self.yaw_invert_check)

        yaw_desc = QtWidgets.QLabel(
            "Flight Yaw Degree = Drone direction (Exif)\n"
            "Gimbal Yaw Degree = Gimbal direction (Exif)"
        )
        yaw_desc.setStyleSheet("font-size: 9pt; color: #666;")
        yaw_layout.addWidget(yaw_desc)

        yaw_group.setLayout(yaw_layout)
        layout.addWidget(yaw_group)

        # Keyboard shortcuts help + Exif stats
        help_group = QtWidgets.QGroupBox("Keyboard Shortcuts")
        help_layout = QtWidgets.QVBoxLayout()
        help_label = QtWidgets.QLabel(
            "H: Altitude +0.2m\n"
            "J: Altitude -0.2m\n"
            "K: Rotation +2°\n"
            "L: Rotation -2°"
        )
        help_label.setStyleSheet("font-size: 9pt;")
        help_layout.addWidget(help_label)
        help_group.setLayout(help_layout)
        # Keep full shortcut text visible.
        help_group.setMinimumHeight(help_group.sizeHint().height())

        exif_group = QtWidgets.QGroupBox("Exif Info")
        exif_layout = QtWidgets.QVBoxLayout()
        self.exif_stats_label = QtWidgets.QLabel("")
        self.exif_stats_label.setStyleSheet("font-size: 9pt; color: #666;")
        exif_layout.addWidget(self.exif_stats_label)
        exif_group.setLayout(exif_layout)

        info_stack = QtWidgets.QWidget()
        info_stack_layout = QtWidgets.QVBoxLayout(info_stack)
        info_stack_layout.setContentsMargins(0, 0, 0, 0)
        info_stack_layout.setSpacing(6)
        info_stack_layout.addWidget(help_group)
        info_stack_layout.addWidget(exif_group)
        layout.addWidget(info_stack)

        # Buttons
        btn_revert = QtWidgets.QPushButton("Revert")
        btn_revert.clicked.connect(self._on_revert)
        layout.addWidget(btn_revert)
        
        btn_save = QtWidgets.QPushButton("Save")
        btn_save.clicked.connect(self._on_save)
        layout.addWidget(btn_save)
        
        layout.addStretch()
        
        self._update_altitude_reference_labels()
        self._update_exif_stats_label()
        return panel

    def _update_altitude_reference_labels(self) -> None:
        try:
            rel_vals = [float(m.alt_m) for m in self.metas]
            rel_avg = float(np.mean(rel_vals))
            rel_min = float(np.min(rel_vals))
            rel_max = float(np.max(rel_vals))
        except Exception:
            rel_avg = 0.0
            rel_min = 0.0
            rel_max = 0.0

        corr = float(self.alt_correction_m_gui)
        cam_water_avg = rel_avg + corr

        if hasattr(self, "cam_water_avg_label"):
            self.cam_water_avg_label.setText(f"Avg. Camera-Water Distance: {cam_water_avg:.2f} m")
        if hasattr(self, "rel_alt_avg_label"):
            self.rel_alt_avg_label.setText(f"Avg: {rel_avg:.2f} m")
        if hasattr(self, "rel_alt_min_label"):
            self.rel_alt_min_label.setText(f"Min: {rel_min:.2f} m")
        if hasattr(self, "rel_alt_max_label"):
            self.rel_alt_max_label.setText(f"Max: {rel_max:.2f} m")

    def _update_exif_stats_label(self) -> None:
        if not hasattr(self, "exif_stats_label"):
            return
        if not self.metas:
            self.exif_stats_label.setText("No EXIF stats available")
            return

        self.exif_stats_label.setText("\n".join(_build_ui_info(self.metas, self._get_current_options())["exif_lines"]))

    def _refresh_metas_for_current_yaw(self) -> None:
        """Reload PhotoMeta list with current yaw rules.

        This is required because yaw_mode/yaw_invert change how yaw is interpreted
        during EXIF extraction (meta.yaw_deg).
        """
        self.metas = self._load_metas()
        if not self.metas:
            return

        # Recompute canvas geometry (world extents may change slightly with yaw/alt-correction)
        current_opts = self._get_current_options()
        self.origin_lat, self.origin_lon, self.min_x, self.min_y, self.max_x, self.max_y, self.mpp = _compute_canvas(
            self.metas, current_opts
        )
        self.canvas_w = int(math.ceil((self.max_x - self.min_x) / self.mpp))
        self.canvas_h = int(math.ceil((self.max_y - self.min_y) / self.mpp))

        # Recompute preview geometry
        self._preview_max_dim = max(64, int(getattr(self.options, "preview_max_dim", 2048)))
        self._preview_scale = min(1.0, self._preview_max_dim / max(self.canvas_w, self.canvas_h))
        self._preview_w = int(self.canvas_w * self._preview_scale)
        self._preview_h = int(self.canvas_h * self._preview_scale)
        self._preview_mpp = self.mpp / self._preview_scale

        self._update_altitude_reference_labels()
        self._update_exif_stats_label()

    def _on_yaw_mode_changed(self, index: int):
        """Handle yaw mode combo box change. Regenerate preview layers."""
        old_mode = self.yaw_mode
        if index == 0:
            self.yaw_mode = "both"
        elif index == 1:
            self.yaw_mode = "flight_only"
        elif index == 2:
            self.yaw_mode = "gimbal_only"

        if old_mode != self.yaw_mode:
            sys.stderr.write(f"Yaw mode changed to: {self.yaw_mode}\n")
            # Reload metas because yaw interpretation changes yaw_deg
            self._refresh_metas_for_current_yaw()
            self._regenerate_preview_only(auto_fit=False)

    def _on_yaw_invert_changed(self, state: int):
        """Handle yaw invert checkbox change. Regenerate preview layers."""
        old_invert = self.yaw_invert
        self.yaw_invert = (state == QtCore.Qt.CheckState.Checked.value)

        if old_invert != self.yaw_invert:
            sys.stderr.write(f"Yaw invert changed to: {self.yaw_invert}\n")
            # Reload metas because yaw interpretation changes yaw_deg
            self._refresh_metas_for_current_yaw()
            self._regenerate_preview_only(auto_fit=False)

    def _update_preview_geometry_for_current_options(self) -> None:
        """Recompute preview canvas geometry for current options.

        This prevents the preview mosaic from being implicitly clipped by an outdated
        preview canvas size when alt-correction/yaw-offset changes.
        """
        current_opts = self._get_current_options()
        self.origin_lat, self.origin_lon, self.min_x, self.min_y, self.max_x, self.max_y, self.mpp = _compute_canvas(
            self.metas, current_opts
        )
        self.canvas_w = int(math.ceil((self.max_x - self.min_x) / self.mpp))
        self.canvas_h = int(math.ceil((self.max_y - self.min_y) / self.mpp))

        self._preview_max_dim = max(64, int(getattr(self.options, "preview_max_dim", 2048)))
        self._preview_scale = min(1.0, self._preview_max_dim / max(self.canvas_w, self.canvas_h))
        self._preview_w = int(self.canvas_w * self._preview_scale)
        self._preview_h = int(self.canvas_h * self._preview_scale)
        self._preview_mpp = self.mpp / self._preview_scale

    def _regenerate_preview_only(self, auto_fit: bool = False):
        """Regenerate preview using cached low-res sources; do not re-read originals.

        auto_fit:
          - False (default): preserve current view transform (no fit reset)
          - True: fit view to scene if scene size changed (use for startup/revert)
        """
        old_size = getattr(self, "_last_scene_size", None)
        self._update_preview_geometry_for_current_options()

        self.preview_image = self._generate_preview()
        self._update_display()

        new_size = (int(self.preview_image.width), int(self.preview_image.height)) if self.preview_image is not None else None
        self._last_scene_size = new_size

        if auto_fit and new_size is not None and new_size != old_size:
            self._fit_view_to_scene()

    def _fit_view_to_scene(self) -> None:
        """Fit the view to the current scene rect."""
        if not hasattr(self, "view"):
            return
        if not hasattr(self, "scene"):
            return
        rect = self.scene.sceneRect()
        if rect.isNull():
            return
        # Keep aspect ratio and fit to the available viewport size
        self.view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _on_revert(self):
        self.alt_correction_m_gui = float(self.options.alt_correction_m)
        self.yaw_offset_deg_gui = float(self.options.yaw_offset_deg)
        if hasattr(self, "alt_correction_input"):
            self.alt_correction_input.setText(f"{self.alt_correction_m_gui:.2f}")
        if hasattr(self, "yaw_offset_input"):
            self.yaw_offset_input.setText(f"{self.yaw_offset_deg_gui:.2f}")
        self._update_altitude_reference_labels()
        self._regenerate_preview_only(auto_fit=True)
        self.statusBar().showMessage("Reverted to original")

    def _on_save(self):
        """Save: read originals and warp at full resolution only here."""
        progress = QtWidgets.QProgressDialog(
            "Generating full-resolution mosaic...",
            "Cancel",
            0,
            len(self.metas) + 1,
            self,
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            progress.setLabelText("Releasing preview memory...")
            QtWidgets.QApplication.processEvents()
            self.statusBar().showMessage("Releasing preview memory before save...")
            self._release_preview_memory()
            self._show_preview_placeholder("Saving in progress...\nPreview is temporarily unavailable.")
            QtWidgets.QApplication.processEvents()

            save_opts = self._get_current_options()

            # Recompute full-res canvas with current options
            self.origin_lat, self.origin_lon, self.min_x, self.min_y, self.max_x, self.max_y, self.mpp = _compute_canvas(self.metas, save_opts)
            self.canvas_w = int(math.ceil((self.max_x - self.min_x) / self.mpp))
            self.canvas_h = int(math.ceil((self.max_y - self.min_y) / self.mpp))
            _ensure_output_dimension_limit(self.out_path, self.canvas_w, self.canvas_h)
            _ensure_save_memory_headroom(self.canvas_w, self.canvas_h, self.out_path, self.metas, save_opts, self.mpp)

            base = Image.new("RGBA", (self.canvas_w, self.canvas_h), (0, 0, 0, 0))
            metas_sorted = sorted(self.metas, key=lambda m: _effective_alt_m(m, save_opts), reverse=True)
            use_crop_opt = bool(getattr(save_opts, "crop_optimize", False))
            best_dist2: Optional[np.ndarray] = None
            if use_crop_opt:
                best_dist2 = np.full((self.canvas_h, self.canvas_w), np.inf, dtype=np.float32)

            for i, meta in enumerate(metas_sorted):
                if progress.wasCanceled():
                    self.statusBar().showMessage("Save cancelled")
                    return

                progress.setValue(i)
                progress.setLabelText(f"Processing {i+1}/{len(metas_sorted)}: {os.path.basename(meta.path)}")
                QtWidgets.QApplication.processEvents()

                _ensure_memory_headroom(
                    _estimate_photo_peak_bytes(meta, self.canvas_w, self.canvas_h, self.mpp, save_opts),
                    context=f"processing {os.path.basename(meta.path)}",
                    margin_bytes=MEMORY_SAFETY_MARGIN_BYTES // 2,
                )

                warped, offset = _warp_photo_to_canvas(
                    meta,
                    origin_lat=self.origin_lat,
                    origin_lon=self.origin_lon,
                    min_x=self.min_x,
                    max_y=self.max_y,
                    mpp_out=self.mpp,
                    canvas_w=self.canvas_w,
                    canvas_h=self.canvas_h,
                    options=save_opts,
                )
                if use_crop_opt and best_dist2 is not None:
                    cu, cv = _photo_center_canvas_xy(
                        meta,
                        origin_lat=self.origin_lat,
                        origin_lon=self.origin_lon,
                        min_x=self.min_x,
                        max_y=self.max_y,
                        mpp_out=self.mpp,
                        options=save_opts,
                    )
                    _composite_nearest_center_rgba_inplace(base, best_dist2, warped, offset, cu, cv)
                else:
                    _alpha_blend_rgba_over_rgba_inplace(base, warped, offset)
                del warped

            progress.setValue(len(metas_sorted))
            progress.setRange(0, 0)
            progress.setLabelText("Saving... preparing metrics")
            progress.setCancelButton(None)
            QtWidgets.QApplication.processEvents()

            out_dir = os.path.dirname(os.path.abspath(self.out_path))
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            save_state = SaveProgressState()
            save_result: Dict[str, Optional[BaseException]] = {"error": None}

            def _save_worker() -> None:
                try:
                    _save_image_job(base, self.out_path, save_opts, save_state)
                except BaseException as exc:
                    save_result["error"] = exc

            save_thread = threading.Thread(target=_save_worker, daemon=True)
            save_loop = QtCore.QEventLoop()
            save_timer = QtCore.QTimer(self)
            save_timer.setInterval(500)

            def _update_save_ui() -> None:
                msg = _format_save_progress_text(save_state)
                progress.setLabelText(msg.replace(", ", " / "))
                self.statusBar().showMessage(msg)
                if not save_thread.is_alive():
                    save_timer.stop()
                    save_loop.quit()

            save_timer.timeout.connect(_update_save_ui)
            save_thread.start()
            _update_save_ui()
            save_timer.start()
            save_loop.exec()
            save_thread.join()

            if save_result["error"] is not None:
                raise save_result["error"]

            saved_path = os.path.abspath(self.out_path)
            self.statusBar().showMessage(f"Saved to {saved_path}")
            QtWidgets.QMessageBox.information(self, "Success", f"Saved to: {saved_path}")

        except MemoryPressureError as e:
            progress.close()
            self.statusBar().showMessage("Save aborted: insufficient memory")
            QtWidgets.QMessageBox.warning(self, "Insufficient Memory", str(e))
            sys.stderr.write(f"Memory check aborted save: {e}\n")
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
            sys.stderr.write(f"Error during save: {e}\n")
        finally:
            try:
                progress.setLabelText("Restoring preview...")
                self.statusBar().showMessage("Restoring preview... Please wait.")
                self._show_preview_placeholder("Rebuilding preview...\nPlease wait.")
                QtWidgets.QApplication.processEvents()
                self._restore_preview_after_save()
            except Exception as e:
                sys.stderr.write(f"Warning: failed to restore preview after save: {e}\n")
            progress.close()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        if key == QtCore.Qt.Key.Key_H:
            self.alt_correction_m_gui += 0.2
            self.alt_correction_input.setText(f"{self.alt_correction_m_gui:.2f}")
            self._update_altitude_reference_labels()
            self._regenerate_preview_only(auto_fit=False)
            return
        if key == QtCore.Qt.Key.Key_J:
            self.alt_correction_m_gui -= 0.2
            self.alt_correction_input.setText(f"{self.alt_correction_m_gui:.2f}")
            self._update_altitude_reference_labels()
            self._regenerate_preview_only(auto_fit=False)
            return
        if key == QtCore.Qt.Key.Key_K:
            self.yaw_offset_deg_gui += 2.0
            self.yaw_offset_input.setText(f"{self.yaw_offset_deg_gui:.2f}")
            self._regenerate_preview_only(auto_fit=False)
            return
        if key == QtCore.Qt.Key.Key_L:
            self.yaw_offset_deg_gui -= 2.0
            self.yaw_offset_input.setText(f"{self.yaw_offset_deg_gui:.2f}")
            self._regenerate_preview_only(auto_fit=False)
            return
        super().keyPressEvent(event)

    def _init_ui(self) -> None:
        """Initialize Qt widgets/layout."""
        self.setWindowTitle("bvpp - Mosaic GUI")
        self.resize(1400, 900)

        central = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout(central)

        # Graphics view
        self.scene = QtWidgets.QGraphicsScene()
        self.view = MosaicViewer(self)
        self.view.setScene(self.scene)
        vlayout.addWidget(self.view, stretch=1)

        # Controls
        controls = self._create_controls()
        vlayout.addWidget(controls, stretch=0)

        self.setCentralWidget(central)

        # Status bar
        self.setStatusBar(QtWidgets.QStatusBar())

        # First render
        self._update_display()

        # Defer initial fit until after the window has been shown and layout is done.
        self._pending_initial_fit = True
        QtCore.QTimer.singleShot(0, self._fit_view_to_scene)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if getattr(self, "_pending_initial_fit", False):
            self._pending_initial_fit = False
            QtCore.QTimer.singleShot(0, self._fit_view_to_scene)

    def _on_alt_correction_input(self) -> None:
        """Apply altitude correction edit and refresh preview (from cached low-res sources)."""
        try:
            self.alt_correction_m_gui = float(self.alt_correction_input.text())
        except Exception:
            self.alt_correction_input.setText(f"{self.alt_correction_m_gui:.2f}")
            return

        # Altitude correction affects placement/scale => preview must be re-warped (but from cache)
        self._update_altitude_reference_labels()
        self._regenerate_preview_only(auto_fit=False)

    def _on_yaw_offset_input(self) -> None:
        """Apply yaw_offset (deg) edit and refresh preview (from cached low-res sources)."""
        try:
            self.yaw_offset_deg_gui = float(self.yaw_offset_input.text())
        except Exception:
            self.yaw_offset_input.setText(f"{self.yaw_offset_deg_gui:.2f}")
            return

        # yaw_offset affects per-image rotation => preview must be re-warped (but from cache)
        self._regenerate_preview_only(auto_fit=False)

    def _on_opacity_changed(self, value: int) -> None:
        """Preview-only opacity change."""
        self.options = RenderOptions(
            undistort=self.options.undistort,
            k1=self.options.k1,
            k2=self.options.k2,
            k3=self.options.k3,
            alt_correction_m=float(self.alt_correction_m_gui),
            yaw_offset_deg=float(self.yaw_offset_deg_gui),
            yaw_invert=self.yaw_invert,
            yaw_both=(self.yaw_mode == "both"),
            yaw_gimbal_only=(self.yaw_mode == "gimbal_only"),
            yaw_flight_only=(self.yaw_mode == "flight_only"),
            opacity_pct=float(value),
            roi_warp=self.options.roi_warp,
            roi_margin_px=self.options.roi_margin_px,
            jpg_quality=self.options.jpg_quality,
            png_compress_level=self.options.png_compress_level,
            preview_max_dim=int(getattr(self.options, "preview_max_dim", 2048)),
            use_pitch=bool(getattr(self.options, "use_pitch", False)),
            crop_optimize=bool(getattr(self.options, "crop_optimize", False)),
            hfov_deg_override=getattr(self.options, "hfov_deg_override", None),
        )
        if hasattr(self, "opacity_label"):
            self.opacity_label.setText(f"{int(value)}%")
        self._regenerate_preview_only(auto_fit=False)


def launch_gui(in_dir: str, out_path: str, options: RenderOptions) -> None:
    """Launch the GUI application."""
    if not QT_AVAILABLE:
        raise SystemExit("PyQt6 not installed. Install with: pip install PyQt6")
    
    app = QtWidgets.QApplication(sys.argv)
    window = MosaicGUI(in_dir, out_path, options)
    window.show()
    sys.exit(app.exec())


# ============================================================================
# End GUI Implementation
# ============================================================================


def main(argv: Optional[List[str]] = None) -> None:
    t0 = time.perf_counter()
    try:
        try:
            args = _parse_args(argv)
            if getattr(args, "inspect_only", False):
                inspect_directory(args.in_dir)
                return
            opts = _render_options_from_args(args)
            if getattr(args, "webui", False):
                launch_webui(
                    args.in_dir,
                    args.out,
                    opts,
                    host=str(getattr(args, "webui_host", "127.0.0.1")),
                    port=int(getattr(args, "webui_port", 8765)),
                    debug=bool(getattr(args, "webui_debug", False)),
                )
                return
            if getattr(args, "gui", False):
                launch_gui(args.in_dir, args.out, opts)
                return
            mosaic(args.in_dir, args.out, opts)
        except MemoryPressureError as e:
            raise SystemExit(str(e))
    finally:
        dt = time.perf_counter() - t0
        sys.stderr.write(f"Elapsed: {dt:.3f} sec\n")


if __name__ == "__main__":
    main()
