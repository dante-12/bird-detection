#!/usr/bin/env python3

"""Bird's-eye Photo Patchwork (bvpp)

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

"""

from __future__ import annotations

import argparse
import gc
import glob
import io
import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union

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

# Optional GUI dependencies (PyQt6)
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
except ImportError:
    QtCore = QtGui = QtWidgets = None

# Pillowで巨大画像を扱う（必要なら制限解除）
Image.MAX_IMAGE_PIXELS = None

SUPPORTED_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
MAX_JPG_TIF_DIM_PX = 64000
MEMORY_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024

# Metadata keys we expect to use (as seen by exifread)
INSPECT_KEYS: Dict[str, Tuple[str, ...]] = {
    "lat": ("GPS Latitude", "XMP DJI:GPSLatitude", "XMP exif:GPSLatitude", "GPS Position"),
    "lon": ("GPS Longitude", "XMP DJI:GPSLongitude", "XMP exif:GPSLongitude", "GPS Position"),
    "alt": ("Relative Altitude", "XMP DJI:RelativeAltitude", "XMP DJI:RelativeAltitudeMeters", "GPS GPSAltitude"),
    "flight_yaw": ("Flight Yaw Degree", "XMP DJI:FlightYawDegree", "XMP DJI:FlightYaw"),
    "gimbal_yaw": ("Gimbal Yaw Degree", "XMP DJI:GimbalYawDegree", "XMP DJI:GimbalYaw"),
    "gimbal_pitch": ("Gimbal Pitch Degree", "XMP DJI:GimbalPitchDegree", "XMP DJI:GimbalPitch"),
    "flight_pitch": ("Flight Pitch Degree", "XMP DJI:FlightPitchDegree", "XMP DJI:FlightPitch"),
    "flight_roll": ("Flight Roll Degree", "XMP DJI:FlightRollDegree", "XMP DJI:FlightRoll"),
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
    flight_yaw_deg: Optional[float] = None
    gimbal_yaw_deg: Optional[float] = None
    pitch_deg: float = -90.0  # degrees; -90 is nadir
    flight_pitch_deg: Optional[float] = None
    gimbal_pitch_deg: Optional[float] = None
    gimbal_roll_deg: Optional[float] = None
    flight_roll_deg: Optional[float] = None
    product_name: Optional[str] = None
    unique_camera_model: Optional[str] = None


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


class MemoryPressureError(RuntimeError):
    """Raised when the estimated memory demand is too close to the system limit."""


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
        "DejaVuSansMono.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
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
    base_size = max(12, min(64, min(img.width, img.height) // 48))
    font_size = base_size
    font = _load_overlay_font(font_size)
    text_w = text_h = 0
    margin = padding = spacing = 0

    while True:
        font = _load_overlay_font(font_size)
        margin = max(8, font_size // 2)
        padding = max(6, font_size // 2)
        spacing = max(2, font_size // 4)
        text_w, text_h = _measure_overlay_text(draw, lines, font, spacing)
        box_w = text_w + (padding * 2)
        box_h = text_h + (padding * 2)
        if (
            font_size <= 12
            or (box_w + margin <= img.width and box_h + margin <= img.height)
        ):
            break
        font_size = max(12, font_size - 2)

    x0 = margin
    y0 = margin
    x1 = min(img.width - 1, x0 + text_w + (padding * 2))
    y1 = min(img.height - 1, y0 + text_h + (padding * 2))
    if x1 <= x0 or y1 <= y0:
        return

    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, 208), outline=(0, 0, 0, 208))
    draw.multiline_text(
        (x0 + padding, y0 + padding),
        "\n".join(lines),
        fill=(0, 0, 0, 255),
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


def _system_mem_available_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        return None
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

    # Altitude
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
    yaw = _extract_yaw(tags, options)
    pitch = _extract_pitch(tags)
    flight_yaw = _extract_flight_yaw(tags)
    gimbal_yaw = _extract_gimbal_yaw(tags)
    flight_pitch = _extract_flight_pitch(tags)
    flight_roll = _extract_flight_roll(tags)
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

    hfov = _extract_fov(tags, w, h)

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
        yaw_deg=float(yaw) % 360.0,
        flight_yaw_deg=float(flight_yaw) % 360.0 if flight_yaw is not None else None,
        gimbal_yaw_deg=float(gimbal_yaw) % 360.0 if gimbal_yaw is not None else None,
        hfov_deg=float(hfov),
        pitch_deg=float(pitch) if pitch is not None else -90.0,
        flight_pitch_deg=float(flight_pitch) if flight_pitch is not None else None,
        gimbal_pitch_deg=float(gimbal_pitch) if gimbal_pitch is not None else None,
        gimbal_roll_deg=float(gimbal_roll) if gimbal_roll is not None else None,
        flight_roll_deg=float(flight_roll) if flight_roll is not None else None,
        product_name=product_name,
        unique_camera_model=unique_camera_model,
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
        "--preview-max-pix",
        type=int,
        default=2048,
        help="GUI preview max width/height in pixels (default: 2048)",
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
        
        if QtWidgets is None:
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
                yaw_deg=meta.yaw_deg,
                hfov_deg=meta.hfov_deg,
                pitch_deg=meta.pitch_deg,
                flight_pitch_deg=meta.flight_pitch_deg,
                gimbal_pitch_deg=meta.gimbal_pitch_deg,
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

        def _stats(values: List[Optional[float]]) -> str:
            nums = [float(v) for v in values if v is not None]
            if not nums:
                return "N/A"
            return f"avg={float(np.mean(nums)):.2f}, min={float(np.min(nums)):.2f}, max={float(np.max(nums)):.2f}"

        def _first_text(values: List[Optional[str]]) -> str:
            for v in values:
                if v is not None:
                    s = str(v).strip()
                    if s:
                        return s
            return "N/A"

        fov_vals = [float(m.hfov_deg) for m in self.metas]
        fov_avg = float(np.mean(fov_vals))
        fov_min = float(np.min(fov_vals))
        fov_max = float(np.max(fov_vals))
        if (fov_max - fov_min) <= 1e-3:
            fov_line = f"FOV [deg]: {fov_avg:.2f} (uniform)"
        else:
            fov_line = f"FOV [deg]: mismatch detected (avg={fov_avg:.2f}, min={fov_min:.2f}, max={fov_max:.2f})"

        fp_line = "Flight Pitch [deg]: " + _stats([getattr(m, "flight_pitch_deg", None) for m in self.metas])
        gp_line = "Gimbal Pitch [deg]: " + _stats([getattr(m, "gimbal_pitch_deg", None) for m in self.metas])
        gr_line = "Gimbal Roll [deg]: " + _stats([getattr(m, "gimbal_roll_deg", None) for m in self.metas])
        fr_line = "Flight Roll [deg]: " + _stats([getattr(m, "flight_roll_deg", None) for m in self.metas])
        fy_line = "Flight Yaw [deg]: " + _stats([getattr(m, "flight_yaw_deg", None) for m in self.metas])
        gy_line = "Gimbal Yaw [deg]: " + _stats([getattr(m, "gimbal_yaw_deg", None) for m in self.metas])
        product_line = "Product Name: " + _first_text([getattr(m, "product_name", None) for m in self.metas])
        model_line = "Unique Camera Model: " + _first_text([getattr(m, "unique_camera_model", None) for m in self.metas])
        self.exif_stats_label.setText(
            "\n".join([product_line, model_line, fov_line, fp_line, gp_line, gr_line, fr_line, fy_line, gy_line])
        )

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
        )
        if hasattr(self, "opacity_label"):
            self.opacity_label.setText(f"{int(value)}%")
        self._regenerate_preview_only(auto_fit=False)


def launch_gui(in_dir: str, out_path: str, options: RenderOptions) -> None:
    """Launch the GUI application."""
    if QtWidgets is None:
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
            if getattr(args, "gui", False):
                launch_gui(args.in_dir, args.out, RenderOptions(
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
                ))
                return
            opts = RenderOptions(
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
            )
            mosaic(args.in_dir, args.out, opts)
        except MemoryPressureError as e:
            raise SystemExit(str(e))
    finally:
        dt = time.perf_counter() - t0
        sys.stderr.write(f"Elapsed: {dt:.3f} sec\n")


if __name__ == "__main__":
    main()
