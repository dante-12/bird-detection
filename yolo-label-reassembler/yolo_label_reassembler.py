#!/usr/bin/env python3
# Author: Naoki

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, QPointF, QRectF, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QKeyEvent, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


try:
    import pyvips  # type: ignore
except Exception:
    pyvips = None


@dataclass
class DetectionRecord:
    class_name: str
    class_id: str
    confidence: float
    x1: float
    x2: float
    y1: float
    y2: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_orig", help="path to original large image")
    p.add_argument("--in_yolo", required=True, help="path to preview large image (YOLO labeled)")
    p.add_argument("--in_csv", help="path to detection csv")
    p.add_argument("--out_dir_images", default=".", help="output directory for saved images (default: current directory)")
    p.add_argument("--out_dir_labels", default=".", help="output directory for saved label txt files (default: current directory)")
    return p.parse_args()


def resolve_output_dir(path: str, role: str) -> Path:
    p = Path(path)
    if p.exists() and p.is_dir():
        return p
    print(
        f"Warning: {role} directory not found: {path}. Falling back to current directory.",
        file=sys.stderr,
    )
    return Path(".").resolve()


def require_pyvips_and_libvips() -> None:
    if pyvips is None:
        print(
            "Error: pyvips is not available. Install both libvips and pyvips before running this tool.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        # Touch both Python binding and native libvips entry points.
        _ = pyvips.version(0), pyvips.version(1), pyvips.version(2)
        _ = pyvips.Image.black(1, 1)
    except Exception as e:
        print(
            f"Error: pyvips is installed but libvips is not usable: {e}",
            file=sys.stderr,
        )
        print(
            "Please install libvips runtime/devel packages and pyvips, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)


class StartupProgress:
    def __init__(self, total_units: int):
        self.total_units = max(1, total_units)
        self.current = 0.0
        self.width = 30
        self._pulse_phase = 0

    def update(self, message: str, units: float = 1.0) -> None:
        self.current = min(float(self.total_units), self.current + max(0.0, units))
        ratio = self.current / float(self.total_units)
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        pct = int(round(ratio * 100))
        sys.stderr.write(f"\r[{bar}] {pct:3d}% {message: <50}")
        sys.stderr.flush()
        time.sleep(0.015)

    def pulse(self, message: str) -> None:
        ratio = self.current / float(self.total_units)
        base_filled = int(round(self.width * ratio))
        pulse_pos = base_filled + (self._pulse_phase % 4)
        self._pulse_phase += 1
        slots = ["-"] * self.width
        for i in range(min(base_filled, self.width)):
            slots[i] = "#"
        if 0 <= pulse_pos < self.width:
            slots[pulse_pos] = ">"
        bar = "".join(slots)
        pct = int(round(ratio * 100))
        sys.stderr.write(f"\r[{bar}] {pct:3d}% {message: <50}")
        sys.stderr.flush()
        time.sleep(0.04)

    def finish(self, message: str = "Ready") -> None:
        self.current = float(self.total_units)
        self.update(message, units=0.0)
        sys.stderr.write("\n")
        sys.stderr.flush()


def read_image_size(path: str) -> Tuple[int, int]:
    if pyvips is not None:
        im = pyvips.Image.new_from_file(path, access="sequential")
        return int(im.width), int(im.height)
    from PIL import Image

    # This tool intentionally handles very large images.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def load_detection_csv(path: str) -> List[DetectionRecord]:
    required = {"class", "class_id", "confidence", "x1", "x2", "y1", "y2", "bbox_w", "bbox_h"}
    out: List[DetectionRecord] = []
    bad_rows = 0

    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fields = set(rdr.fieldnames)
        missing = required - fields
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in rdr:
            try:
                rec = DetectionRecord(
                    class_name=(row.get("class") or "unknown").strip() or "unknown",
                    class_id=(row.get("class_id") or "").strip(),
                    confidence=float(row["confidence"]),
                    x1=float(row["x1"]),
                    x2=float(row["x2"]),
                    y1=float(row["y1"]),
                    y2=float(row["y2"]),
                )
                if rec.x2 < rec.x1:
                    rec.x1, rec.x2 = rec.x2, rec.x1
                if rec.y2 < rec.y1:
                    rec.y1, rec.y2 = rec.y2, rec.y1
                out.append(rec)
            except Exception:
                bad_rows += 1

    if bad_rows:
        print(f"Warning: skipped {bad_rows} invalid CSV rows.", file=sys.stderr)
    return out


def _vips_to_qimage(vimg, progress_cb=None, progress_prefix: str = "") -> QImage:
    bands = int(vimg.bands)
    width = int(vimg.width)
    height = int(vimg.height)
    fmt = str(vimg.format)
    if fmt != "uchar":
        if progress_cb:
            progress_cb(f"{progress_prefix} cast to 8-bit", True)
        vimg = vimg.cast("uchar")
        fmt = "uchar"
    if bands == 1:
        qfmt = QImage.Format_Grayscale8
    elif bands == 3:
        qfmt = QImage.Format_RGB888
    else:
        if bands > 4:
            if progress_cb:
                progress_cb(f"{progress_prefix} reducing channels", True)
            vimg = vimg.extract_band(0, n=4)
            bands = 4
        elif bands == 2:
            if progress_cb:
                progress_cb(f"{progress_prefix} expanding channels", True)
            vimg = vimg.bandjoin(vimg[0])
            bands = 3
        qfmt = QImage.Format_RGBA8888 if bands == 4 else QImage.Format_RGB888
    if progress_cb:
        progress_cb(f"{progress_prefix} exporting pixel buffer", True)
    raw = _run_with_pulse(
        fn=vimg.write_to_memory,
        progress_cb=progress_cb,
        message=f"{progress_prefix} exporting pixel buffer",
        poll_sec=0.12,
    )
    stride = width * bands
    if progress_cb:
        progress_cb(f"{progress_prefix} building Qt image", True)
    return QImage(raw, width, height, stride, qfmt).copy()


def _run_with_pulse(fn, progress_cb=None, message: str = "", poll_sec: float = 0.1):
    result_box = {"ok": False, "result": None, "error": None}

    def _runner():
        try:
            result_box["result"] = fn()
            result_box["ok"] = True
        except Exception as e:
            result_box["error"] = e

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    while th.is_alive():
        if progress_cb is not None:
            progress_cb(f"{message} (working)", False)
        th.join(timeout=poll_sec)
    if result_box["error"] is not None:
        raise result_box["error"]
    return result_box["result"]


def load_preview_qimage(path: str, max_dim: int = 6000, progress_cb=None) -> Tuple[QImage, int, int]:
    if pyvips is not None:
        if progress_cb:
            progress_cb("Preparing preview image... opening file", True)
        vimg = pyvips.Image.new_from_file(path, access="sequential")
        orig_w = int(vimg.width)
        orig_h = int(vimg.height)
        if progress_cb:
            progress_cb("Preparing preview image... computing scale", True)
        scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
        if scale < 1.0:
            if progress_cb:
                progress_cb("Preparing preview image... resizing", True)
            vimg = vimg.resize(scale, kernel="linear")
        if progress_cb:
            progress_cb("Preparing preview image... converting to GUI image", True)
        qimg = _vips_to_qimage(vimg, progress_cb=progress_cb, progress_prefix="Preparing preview image...")
        if progress_cb:
            progress_cb("Preparing preview image... finalizing", True)
        return qimg, orig_w, orig_h

    from PIL import Image

    # This tool intentionally handles very large images.
    Image.MAX_IMAGE_PIXELS = None
    if progress_cb:
        progress_cb("Preparing preview image... opening file", True)
    with Image.open(path) as pil:
        if progress_cb:
            progress_cb("Preparing preview image... decoding", True)
        pil = pil.convert("RGB")
        orig_w, orig_h = pil.size
        if progress_cb:
            progress_cb("Preparing preview image... computing scale", True)
        scale = min(1.0, float(max_dim) / float(max(orig_w, orig_h)))
        if scale < 1.0:
            if progress_cb:
                progress_cb("Preparing preview image... downscaling", True)
            pil.thumbnail((int(orig_w * scale), int(orig_h * scale)))
        if progress_cb:
            progress_cb("Preparing preview image... transferring pixels", True)
        data = _run_with_pulse(
            fn=lambda: pil.tobytes("raw", "RGB"),
            progress_cb=progress_cb,
            message="Preparing preview image... transferring pixels",
            poll_sec=0.12,
        )
        if progress_cb:
            progress_cb("Preparing preview image... creating Qt image", True)
        qim = QImage(data, pil.width, pil.height, pil.width * 3, QImage.Format_RGB888).copy()
        if progress_cb:
            progress_cb("Preparing preview image... finalizing", True)
        return qim, orig_w, orig_h


def _pick_lod(src_w: int, src_h: int, out_w: int, out_h: int) -> int:
    ratio = min(src_w / max(1.0, float(out_w)), src_h / max(1.0, float(out_h)))
    lod = 1
    while lod * 2 <= ratio:
        lod *= 2
    return lod


class HighDetailWorker(QObject):
    ready = Signal(int, object, object)
    failed = Signal(int, str)
    progress = Signal(int, int, int, str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self._vips_image = None
        self._pending: Optional[Tuple[int, Tuple[int, int, int, int, int, int, int]]] = None
        self._running = False
        self._active_request_id: Optional[int] = None

    def _emit_progress(self, step: int, total: int, message: str) -> None:
        if self._active_request_id is None:
            return
        self.progress.emit(self._active_request_id, step, total, message)

    def _ensure_vips(self) -> None:
        if pyvips is None:
            return
        if self._vips_image is None:
            # Keep opened image for repeated crop requests.
            self._vips_image = pyvips.Image.new_from_file(self.path, access="random")

    def _load_region(self, payload: Tuple[int, int, int, int, int, int, int]) -> QImage:
        x, y, width, height, out_w, out_h, lod = payload
        width = max(1, width)
        height = max(1, height)
        out_w = max(1, out_w)
        out_h = max(1, out_h)
        lod = max(1, lod)

        if pyvips is not None:
            total = 6
            self._emit_progress(1, total, "opening image")
            self._ensure_vips()
            self._emit_progress(2, total, "cropping region")
            region = self._vips_image.crop(x, y, width, height)
            if lod > 1:
                self._emit_progress(3, total, f"downscaling (lod={lod})")
                region = region.resize(1.0 / float(lod), kernel="linear")
            else:
                self._emit_progress(3, total, "downscaling skipped")
            cur_w = max(1, int(region.width))
            cur_h = max(1, int(region.height))
            sx = out_w / float(cur_w)
            sy = out_h / float(cur_h)
            self._emit_progress(4, total, "resizing tile")
            region = region.resize(sx, vscale=sy, kernel="linear")
            self._emit_progress(5, total, "converting to Qt image")
            qimg = _vips_to_qimage(region)
            self._emit_progress(6, total, "finalizing")
            return qimg

        from PIL import Image

        total = 6
        self._emit_progress(1, total, "opening image")
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(self.path) as im:
            self._emit_progress(2, total, "cropping region")
            patch = im.crop((x, y, x + width, y + height)).convert("RGB")
            if lod > 1:
                self._emit_progress(3, total, f"downscaling (lod={lod})")
                down_w = max(1, width // lod)
                down_h = max(1, height // lod)
                patch = patch.resize((down_w, down_h), Image.Resampling.BOX)
            else:
                self._emit_progress(3, total, "downscaling skipped")
            self._emit_progress(4, total, "resizing tile")
            patch = patch.resize((out_w, out_h), Image.Resampling.BILINEAR)
            self._emit_progress(5, total, "transferring pixels")
            data = patch.tobytes("raw", "RGB")
            qimg = QImage(data, patch.width, patch.height, patch.width * 3, QImage.Format_RGB888).copy()
            self._emit_progress(6, total, "finalizing")
            return qimg

    @Slot(int, object)
    def request(self, request_id: int, payload_obj: object) -> None:
        payload = payload_obj  # type: ignore[assignment]
        self._pending = (request_id, payload)
        if not self._running:
            self._running = True
            QTimer.singleShot(0, self._process_next)

    @Slot()
    def _process_next(self) -> None:
        if self._pending is None:
            self._running = False
            return
        request_id, payload = self._pending
        self._active_request_id = request_id
        self._pending = None
        try:
            qimg = self._load_region(payload)
            self.ready.emit(request_id, payload, qimg)
        except Exception as e:
            self.failed.emit(request_id, str(e))
        self._active_request_id = None
        QTimer.singleShot(0, self._process_next)


class EditableRectItem(QGraphicsRectItem):
    HANDLE_SIZE = 16.0

    def __init__(self, rect: QRectF, color: QColor):
        super().__init__(rect)
        self.base_color = color
        self.setPen(self._make_pen(color))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setZValue(10)
        self.editable = True
        self._view_scale = 1.0
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

    def _make_pen(self, color: QColor) -> QPen:
        pen = QPen(color, 1)
        pen.setCosmetic(True)  # Keep 1px regardless of zoom.
        return pen

    def set_color(self, color: QColor) -> None:
        self.base_color = color
        self.setPen(self._make_pen(color))
        self.update()

    def set_editable(self, enabled: bool) -> None:
        self.editable = enabled
        if not enabled:
            self.setSelected(False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, enabled)
        self.update()

    def set_view_scale(self, scale: float) -> None:
        self._view_scale = max(0.01, float(scale))
        self.update()

    def _handles(self) -> dict[str, QRectF]:
        r = self.rect()
        size_scene = self.HANDLE_SIZE / self._view_scale
        hs = size_scene / 2.0
        cx = (r.left() + r.right()) / 2.0
        cy = (r.top() + r.bottom()) / 2.0
        return {
            "tl": QRectF(r.left() - hs, r.top() - hs, size_scene, size_scene),
            "tm": QRectF(cx - hs, r.top() - hs, size_scene, size_scene),
            "tr": QRectF(r.right() - hs, r.top() - hs, size_scene, size_scene),
            "ml": QRectF(r.left() - hs, cy - hs, size_scene, size_scene),
            "mr": QRectF(r.right() - hs, cy - hs, size_scene, size_scene),
            "bl": QRectF(r.left() - hs, r.bottom() - hs, size_scene, size_scene),
            "bm": QRectF(cx - hs, r.bottom() - hs, size_scene, size_scene),
            "br": QRectF(r.right() - hs, r.bottom() - hs, size_scene, size_scene),
        }

    def hit_handle(self, pt: QPointF) -> Optional[str]:
        if not (self.editable and self.isSelected()):
            return None
        for name, hrect in self._handles().items():
            if hrect.contains(pt):
                return name
        return None

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if not (self.editable and self.isSelected()):
            return
        handle_pen = QPen(Qt.GlobalColor.white, 1)
        handle_pen.setCosmetic(True)
        painter.setPen(handle_pen)
        painter.setBrush(self.base_color)
        for hrect in self._handles().values():
            painter.drawRect(hrect)


class ImageCanvas(QGraphicsView):
    detail_request = Signal(int, object)
    detail_loading_changed = Signal(bool)
    detail_progress_changed = Signal(int, int, str)
    view_metrics_changed = Signal(str)
    draft_rect_changed = Signal()

    def __init__(self, pixmap: QPixmap, in_yolo_path: str, orig_size: Tuple[int, int], parent=None):
        super().__init__(parent)
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene_obj.addItem(self.pixmap_item)
        self._image_border_item = QGraphicsRectItem(self.pixmap_item.boundingRect())
        self._image_border_item.setPen(QPen(QColor(120, 120, 120), 2))
        self._image_border_item.setBrush(Qt.BrushStyle.NoBrush)
        self._image_border_item.setZValue(3)
        self._image_border_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene_obj.addItem(self._image_border_item)
        self.detail_item = QGraphicsPixmapItem()
        self.detail_item.setZValue(2)
        self.detail_item.setVisible(False)
        self.scene_obj.addItem(self.detail_item)

        # Keep view rendering cheap. Detail layer provides fidelity while zoomed.
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Keep visible margin around the image so users can see exact image bounds.
        self._scene_padding_px = 160.0
        image_rect = self.pixmap_item.boundingRect()
        self.setSceneRect(
            image_rect.adjusted(
                -self._scene_padding_px,
                -self._scene_padding_px,
                self._scene_padding_px,
                self._scene_padding_px,
            )
        )
        self.setBackgroundBrush(QColor(170, 170, 170))

        self.in_yolo_path = in_yolo_path
        self.orig_w, self.orig_h = orig_size
        self.disp_w = pixmap.width()
        self.disp_h = pixmap.height()
        self.scale_x = self.orig_w / max(1.0, float(self.disp_w))
        self.scale_y = self.orig_h / max(1.0, float(self.disp_h))

        self.edit_mode = False
        self.draft_rect: Optional[EditableRectItem] = None
        self.saved_rects: List[EditableRectItem] = []
        self._draft_w_text: Optional[QGraphicsSimpleTextItem] = None
        self._draft_h_text: Optional[QGraphicsSimpleTextItem] = None

        self._drawing = False
        self._draw_start = QPointF()
        self._resizing = False
        self._resize_handle: Optional[str] = None
        self._resize_origin = QRectF()
        self._edit_interaction_locked = False

        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.timeout.connect(self._update_high_detail)
        self._detail_margin_px = 96
        self._fast_detail_backend = pyvips is not None
        self._detail_zoom_threshold = 1.12
        self._last_detail_key: Optional[Tuple[int, int, int, int, int, int, int]] = None
        self._latest_request_id = 0
        self._inflight_key: Optional[Tuple[int, int, int, int, int, int, int]] = None
        self._active_scene_rect = QRectF()
        self._cache_limit_bytes = (256 if self._fast_detail_backend else 96) * 1024 * 1024
        self._cache_bytes = 0
        self._cache: "OrderedDict[Tuple[int, int, int, int, int, int, int], QPixmap]" = OrderedDict()
        self._detail_frames_loaded = 0

        self._detail_thread = QThread(self)
        self._detail_worker = HighDetailWorker(self.in_yolo_path)
        self._detail_worker.moveToThread(self._detail_thread)
        self.detail_request.connect(self._detail_worker.request)
        self._detail_worker.ready.connect(self._on_detail_ready)
        self._detail_worker.failed.connect(self._on_detail_failed)
        self._detail_worker.progress.connect(self._on_detail_progress)
        self._detail_thread.start()

        self.horizontalScrollBar().valueChanged.connect(self._on_scrolled)
        self.verticalScrollBar().valueChanged.connect(self._on_scrolled)

    def shutdown(self) -> None:
        self._detail_timer.stop()
        if self._detail_thread.isRunning():
            self._detail_thread.quit()
            self._detail_thread.wait(1500)

    def initialize_view(self) -> None:
        # Show about 25% of the original image area in the window.
        # Start from fit-to-window, then zoom in so visible area becomes ~1/4.
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        fit_scale = self.transform().m11()
        target_scale = fit_scale * 2.0
        if fit_scale > 0:
            self.scale(target_scale / fit_scale, target_scale / fit_scale)
        self.centerOn(self.pixmap_item.boundingRect().center())
        self._sync_item_view_scale()
        self._emit_view_metrics()
        self._schedule_detail_update()

    def _on_scrolled(self, _value: int) -> None:
        self._schedule_detail_update()
        self._emit_view_metrics()

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = enabled
        self._apply_drag_mode()
        if self.draft_rect:
            self.draft_rect.set_editable(enabled and not self._edit_interaction_locked)
            if not enabled or self._edit_interaction_locked:
                self.draft_rect.setSelected(False)
        if not enabled:
            self._drawing = False
            self._resizing = False
            self._resize_handle = None
            self.discard_draft_rect()

    def _apply_drag_mode(self) -> None:
        # During save, keep pan enabled even if Edit mode is active.
        if self.edit_mode and not self._edit_interaction_locked:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_edit_interaction_locked(self, locked: bool) -> None:
        self._edit_interaction_locked = locked
        self._apply_drag_mode()
        if self.draft_rect:
            self.draft_rect.set_editable(self.edit_mode and not locked)
            if locked:
                self.draft_rect.setSelected(False)
        if locked:
            self._drawing = False
            self._resizing = False
            self._resize_handle = None

    def discard_draft_rect(self) -> None:
        if self.draft_rect is None:
            return
        self.scene_obj.removeItem(self.draft_rect)
        self.draft_rect = None
        if self._draft_w_text is not None:
            self.scene_obj.removeItem(self._draft_w_text)
            self._draft_w_text = None
        if self._draft_h_text is not None:
            self.scene_obj.removeItem(self._draft_h_text)
            self._draft_h_text = None
        self.draft_rect_changed.emit()

    def _emit_view_metrics(self) -> None:
        # 100% means 1 screen pixel == 1 pixel in the original image.
        zx = self.transform().m11() / max(self.scale_x, 1e-9)
        zy = self.transform().m22() / max(self.scale_y, 1e-9)
        zoom_pct = ((zx + zy) * 0.5) * 100.0
        self.view_metrics_changed.emit(f"{self.orig_w} x {self.orig_h} px  |  Zoom {zoom_pct:.1f}%")

    def _ensure_draft_text_items(self) -> None:
        if self._draft_w_text is None:
            self._draft_w_text = QGraphicsSimpleTextItem()
            self._draft_w_text.setBrush(QColor(255, 0, 0))
            self._draft_w_text.setZValue(20)
            self._draft_w_text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            f = QFont(self._draft_w_text.font())
            size = f.pointSizeF()
            if size <= 0:
                size = 12.0
            f.setPointSizeF(size * 2.0)
            self._draft_w_text.setFont(f)
            self.scene_obj.addItem(self._draft_w_text)
        if self._draft_h_text is None:
            self._draft_h_text = QGraphicsSimpleTextItem()
            self._draft_h_text.setBrush(QColor(255, 0, 0))
            self._draft_h_text.setZValue(20)
            self._draft_h_text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            f = QFont(self._draft_h_text.font())
            size = f.pointSizeF()
            if size <= 0:
                size = 12.0
            f.setPointSizeF(size * 2.0)
            self._draft_h_text.setFont(f)
            self.scene_obj.addItem(self._draft_h_text)

    def _update_draft_metrics_overlay(self) -> None:
        if self.draft_rect is None:
            return
        self._ensure_draft_text_items()
        r = self.draft_rect.rect().normalized()
        bw = max(0, int(round(r.width() * self.scale_x)))
        bh = max(0, int(round(r.height() * self.scale_y)))
        w_text = f"{bw}px"
        h_text = f"{bh}px"
        self._draft_w_text.setText(w_text)
        self._draft_h_text.setText(h_text)

        w_rect = self._draft_w_text.boundingRect()
        h_rect = self._draft_h_text.boundingRect()
        # Text items ignore view transforms, so convert their pixel size to scene units.
        view_scale = max(0.01, self.transform().m11())
        w_scene_w = w_rect.width() / view_scale
        w_scene_h = w_rect.height() / view_scale
        h_scene_w = h_rect.width() / view_scale
        h_scene_h = h_rect.height() / view_scale
        # Keep labels away from resize handles.
        margin_scene = 14.0 / view_scale

        # Top edge center
        wx = r.left() + (r.width() - w_scene_w) * 0.5
        wy = max(0.0, r.top() - w_scene_h - margin_scene)
        # Right edge center
        hx = min(self.pixmap_item.boundingRect().right() - h_scene_w, r.right() + margin_scene)
        hy = r.top() + (r.height() - h_scene_h) * 0.5
        self._draft_w_text.setPos(wx, wy)
        self._draft_h_text.setPos(hx, hy)
        self.draft_rect_changed.emit()

    def _schedule_detail_update(self) -> None:
        # Wait until user stops wheel/pan; avoid flooding decode requests.
        self._detail_timer.start(140 if self._fast_detail_backend else 320)

    def _invalidate_detail_overlay(self) -> None:
        # Drop stale detail layer immediately during viewport size changes.
        self.detail_item.setVisible(False)
        self._last_detail_key = None
        self._active_scene_rect = QRectF()
        self._inflight_key = None
        # Ignore any detail result produced for the previous geometry.
        self._latest_request_id += 1
        self.detail_loading_changed.emit(False)

    def _cache_get(self, key: Tuple[int, int, int, int, int, int, int]) -> Optional[QPixmap]:
        pm = self._cache.get(key)
        if pm is None:
            return None
        self._cache.move_to_end(key)
        return pm

    def _cache_put(self, key: Tuple[int, int, int, int, int, int, int], pixmap: QPixmap) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = pixmap
            return
        size = max(1, pixmap.width()) * max(1, pixmap.height()) * 4
        self._cache[key] = pixmap
        self._cache_bytes += size
        self._cache.move_to_end(key)
        while self._cache and self._cache_bytes > self._cache_limit_bytes:
            old_key, old_pm = self._cache.popitem(last=False)
            self._cache_bytes -= max(1, old_pm.width()) * max(1, old_pm.height()) * 4
            if self._cache_bytes < 0:
                self._cache_bytes = 0

    def _show_detail(self, scene_rect: QRectF, pixmap: QPixmap) -> None:
        self.detail_item.setPixmap(pixmap)
        self.detail_item.setPos(scene_rect.left(), scene_rect.top())
        sx = scene_rect.width() / max(1.0, float(pixmap.width()))
        sy = scene_rect.height() / max(1.0, float(pixmap.height()))
        self.detail_item.setTransform(QTransform.fromScale(sx, sy))
        self.detail_item.setVisible(True)

    def _sync_item_view_scale(self) -> None:
        scale = max(0.01, self.transform().m11())
        if self.draft_rect is not None:
            self.draft_rect.set_view_scale(scale)
        for r in self.saved_rects:
            r.set_view_scale(scale)

    @Slot(int, object, object)
    def _on_detail_ready(self, request_id: int, payload_obj: object, qimg_obj: object) -> None:
        if request_id != self._latest_request_id:
            return
        payload = payload_obj  # type: ignore[assignment]
        qimg = qimg_obj  # type: ignore[assignment]
        key = payload
        self._inflight_key = None
        pixmap = QPixmap.fromImage(qimg)
        self._cache_put(key, pixmap)
        self._show_detail(self._active_scene_rect, pixmap)
        self._detail_frames_loaded += 1
        self.detail_loading_changed.emit(False)

    @Slot(int, str)
    def _on_detail_failed(self, request_id: int, _error: str) -> None:
        if request_id != self._latest_request_id:
            return
        self._inflight_key = None
        self.detail_loading_changed.emit(False)

    @Slot(int, int, int, str)
    def _on_detail_progress(self, request_id: int, step: int, total: int, message: str) -> None:
        if request_id != self._latest_request_id:
            return
        self.detail_progress_changed.emit(step, total, message)

    def _update_high_detail(self) -> None:
        zoom = self.transform().m11()
        if zoom < self._detail_zoom_threshold:
            self.detail_item.setVisible(False)
            self._last_detail_key = None
            self.detail_loading_changed.emit(False)
            return

        view_rect = QRectF(self.mapToScene(self.viewport().rect()).boundingRect())
        img_rect = self.pixmap_item.boundingRect()
        scene_rect = view_rect.intersected(img_rect)
        if scene_rect.isEmpty():
            self.detail_item.setVisible(False)
            self.detail_loading_changed.emit(False)
            return

        margin_x = self._detail_margin_px / max(zoom, 1.0)
        margin_y = self._detail_margin_px / max(zoom, 1.0)
        scene_rect = scene_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y).intersected(img_rect)

        ox1 = int(max(0, math.floor(scene_rect.left() * self.scale_x)))
        oy1 = int(max(0, math.floor(scene_rect.top() * self.scale_y)))
        ox2 = int(min(self.orig_w, math.ceil(scene_rect.right() * self.scale_x)))
        oy2 = int(min(self.orig_h, math.ceil(scene_rect.bottom() * self.scale_y)))
        ow = max(1, ox2 - ox1)
        oh = max(1, oy2 - oy1)
        # Use exact inverse mapping from source pixels to display coords to avoid drift.
        disp_left = ox1 / self.scale_x
        disp_top = oy1 / self.scale_y
        disp_w = ow / self.scale_x
        disp_h = oh / self.scale_y

        raw_out_w = max(1, int(math.ceil(disp_w * zoom)))
        raw_out_h = max(1, int(math.ceil(disp_h * zoom)))
        # First zoom render should be responsive; start smaller then switch to full quality.
        if self._detail_frames_loaded == 0:
            max_out = 1400 if self._fast_detail_backend else 900
        else:
            max_out = 3072 if self._fast_detail_backend else 1400
        # Keep output aspect ratio. Independent clamping can distort and cause
        # visible layer misalignment against the base preview when zoomed.
        shrink = min(1.0, max_out / max(1, raw_out_w), max_out / max(1, raw_out_h))
        out_w = max(1, int(round(raw_out_w * shrink)))
        out_h = max(1, int(round(raw_out_h * shrink)))
        lod = _pick_lod(ow, oh, out_w, out_h)

        key = (ox1, oy1, ow, oh, out_w, out_h, lod)
        if self._last_detail_key == key:
            return
        self._last_detail_key = key
        self._active_scene_rect = QRectF(disp_left, disp_top, disp_w, disp_h)

        cached = self._cache_get(key)
        if cached is not None:
            self._show_detail(self._active_scene_rect, cached)
            self.detail_loading_changed.emit(False)
            return

        if self._inflight_key == key:
            return

        self._latest_request_id += 1
        self._inflight_key = key
        self.detail_loading_changed.emit(True)
        self.detail_request.emit(self._latest_request_id, key)

    def _clip_to_image(self, p: QPointF) -> QPointF:
        rect = self.pixmap_item.boundingRect()
        x = min(max(p.x(), rect.left()), rect.right())
        y = min(max(p.y(), rect.top()), rect.bottom())
        return QPointF(x, y)

    def _rect_from_points(self, p1: QPointF, p2: QPointF) -> QRectF:
        r = QRectF(p1, p2).normalized()
        r = r.intersected(self.pixmap_item.boundingRect())
        return r

    def _ensure_draft(self, rect: QRectF) -> None:
        if self.draft_rect is None:
            self.draft_rect = EditableRectItem(rect, QColor(255, 0, 0))
            self.scene_obj.addItem(self.draft_rect)
        else:
            self.draft_rect.setRect(rect)
        self.draft_rect.set_view_scale(self.transform().m11())
        self.draft_rect.set_editable(self.edit_mode)
        if self.edit_mode:
            self.draft_rect.setSelected(True)
        self._update_draft_metrics_overlay()

    def mousePressEvent(self, event) -> None:
        if self._edit_interaction_locked:
            super().mousePressEvent(event)
            return
        if not self.edit_mode or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        scene_pos = self._clip_to_image(scene_pos)

        if self.draft_rect is not None and self.draft_rect.contains(self.draft_rect.mapFromScene(scene_pos)):
            self.draft_rect.setSelected(True)
            handle = self.draft_rect.hit_handle(self.draft_rect.mapFromScene(scene_pos))
            if handle is not None:
                self._resizing = True
                self._resize_handle = handle
                self._resize_origin = QRectF(self.draft_rect.rect())
                return
            super().mousePressEvent(event)
            return

        self._drawing = True
        self._draw_start = scene_pos
        self._ensure_draft(QRectF(scene_pos, scene_pos))

    def mouseMoveEvent(self, event) -> None:
        if self._edit_interaction_locked:
            super().mouseMoveEvent(event)
            return
        if not self.edit_mode:
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        scene_pos = self._clip_to_image(scene_pos)

        if self._drawing and self.draft_rect is not None:
            self.draft_rect.setRect(self._rect_from_points(self._draw_start, scene_pos))
            self._update_draft_metrics_overlay()
            return

        if self._resizing and self.draft_rect is not None and self._resize_handle is not None:
            r = QRectF(self._resize_origin)
            x = scene_pos.x()
            y = scene_pos.y()
            h = self._resize_handle
            if "l" in h:
                r.setLeft(x)
            if "r" in h:
                r.setRight(x)
            if "t" in h:
                r.setTop(y)
            if "b" in h:
                r.setBottom(y)
            r = r.normalized().intersected(self.pixmap_item.boundingRect())
            self.draft_rect.setRect(r)
            self._update_draft_metrics_overlay()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._edit_interaction_locked:
            super().mouseReleaseEvent(event)
            return
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            if self.draft_rect and (self.draft_rect.rect().width() < 1 or self.draft_rect.rect().height() < 1):
                self.scene_obj.removeItem(self.draft_rect)
                self.draft_rect = None
                self.draft_rect_changed.emit()
            self._schedule_detail_update()
            return

        if self._resizing and event.button() == Qt.MouseButton.LeftButton:
            self._resizing = False
            self._resize_handle = None
            self._schedule_detail_update()
            return

        self._schedule_detail_update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self._sync_item_view_scale()
        self._update_draft_metrics_overlay()
        self._emit_view_metrics()
        self._schedule_detail_update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._invalidate_detail_overlay()
        self._sync_item_view_scale()
        self._update_draft_metrics_overlay()
        self._emit_view_metrics()
        self._schedule_detail_update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            not self._edit_interaction_locked
            and
            self.edit_mode
            and event.key() == Qt.Key.Key_Delete
            and self.draft_rect is not None
            and self.draft_rect.isSelected()
        ):
            self.scene_obj.removeItem(self.draft_rect)
            self.draft_rect = None
            self.draft_rect_changed.emit()
            return
        super().keyPressEvent(event)

    def get_draft_rect_in_orig(self) -> Optional[Tuple[int, int, int, int]]:
        if self.draft_rect is None:
            return None
        r = self.draft_rect.rect().normalized()
        x1 = int(max(0, math.floor(r.left() * self.scale_x)))
        y1 = int(max(0, math.floor(r.top() * self.scale_y)))
        x2 = int(min(self.orig_w, math.ceil(r.right() * self.scale_x)))
        y2 = int(min(self.orig_h, math.ceil(r.bottom() * self.scale_y)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def finalize_current_rect(self) -> None:
        if self.draft_rect is None:
            return
        self.draft_rect.set_color(QColor(255, 140, 0))
        self.draft_rect.set_view_scale(self.transform().m11())
        self.draft_rect.set_editable(False)
        self.saved_rects.append(self.draft_rect)
        self.draft_rect = None
        if self._draft_w_text is not None:
            self.scene_obj.removeItem(self._draft_w_text)
            self._draft_w_text = None
        if self._draft_h_text is not None:
            self.scene_obj.removeItem(self._draft_h_text)
            self._draft_h_text = None


class SaveWorker(QObject):
    progress = Signal(int, str)
    done = Signal(str, str, int)
    failed = Signal(str)

    def __init__(
        self,
        in_orig: str,
        out_png: Path,
        out_txt: Path,
        records: List[DetectionRecord],
        rect: Tuple[int, int, int, int],
    ):
        super().__init__()
        self.in_orig = in_orig
        self.out_png = out_png
        self.out_txt = out_txt
        self.records = records
        self.rect = rect

    def run(self) -> None:
        try:
            self.progress.emit(5, "Preparing save...")
            self.progress.emit(45, "Saving cropped image...")
            save_crop_png(self.in_orig, self.out_png, self.rect)
            self.progress.emit(80, "Writing YOLO labels...")
            n = write_yolo_labels(self.out_txt, self.records, self.rect)
            self.progress.emit(100, "Completed.")
            self.done.emit(str(self.out_png), str(self.out_txt), n)
        except Exception as e:
            self.failed.emit(str(e))


def unique_output_paths(out_dir_images: Path, out_dir_labels: Path, in_orig: str) -> Tuple[Path, Path]:
    stem = Path(in_orig).stem
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name = f"{stem}_{ts}"
    out_png = out_dir_images / f"{base_name}.png"
    out_txt = out_dir_labels / f"{base_name}.txt"
    if not out_png.exists() and not out_txt.exists():
        return out_png, out_txt
    i = 1
    while True:
        name = f"{base_name}_{i}"
        cand_png = out_dir_images / f"{name}.png"
        cand_txt = out_dir_labels / f"{name}.txt"
        if not cand_png.exists() and not cand_txt.exists():
            return cand_png, cand_txt
        i += 1


def save_crop_png(in_orig: str, out_png: Path, rect: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("Invalid crop rectangle.")

    if pyvips is not None:
        im = pyvips.Image.new_from_file(in_orig, access="sequential")
        cropped = im.crop(x1, y1, width, height)
        cropped.write_to_file(str(out_png))
        return

    from PIL import Image

    # This tool intentionally handles very large images.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(in_orig) as im:
        patch = im.crop((x1, y1, x2, y2))
        patch.save(out_png)


def write_yolo_labels(
    out_txt: Path,
    records: List[DetectionRecord],
    crop_rect: Tuple[int, int, int, int],
) -> int:
    rx1, ry1, rx2, ry2 = crop_rect
    cw = float(rx2 - rx1)
    ch = float(ry2 - ry1)
    if cw <= 0 or ch <= 0:
        raise ValueError("Invalid crop size.")

    lines: List[str] = []
    for class_id, lx1, lx2, ly1, ly2 in iter_crop_records_for_export(records, crop_rect):
        bw = lx2 - lx1
        bh = ly2 - ly1

        cx = lx1 + bw / 2.0
        cy = ly1 + bh / 2.0
        cx_n = min(max(cx / cw, 0.0), 1.0)
        cy_n = min(max(cy / ch, 0.0), 1.0)
        bw_n = min(max(bw / cw, 0.0), 1.0)
        bh_n = min(max(bh / ch, 0.0), 1.0)
        lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

    with open(out_txt, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")
    return len(lines)


def iter_crop_records_for_export(
    records: List[DetectionRecord],
    crop_rect: Tuple[int, int, int, int],
):
    rx1, ry1, rx2, ry2 = crop_rect
    for rec in records:
        if not (rec.x1 >= rx1 and rec.x2 <= rx2 and rec.y1 >= ry1 and rec.y2 <= ry2):
            continue
        try:
            class_id = int(float(rec.class_id))
        except Exception:
            continue

        lx1 = rec.x1 - rx1
        lx2 = rec.x2 - rx1
        ly1 = rec.y1 - ry1
        ly2 = rec.y2 - ry1

        bw = lx2 - lx1
        bh = ly2 - ly1
        if bw <= 0 or bh <= 0:
            continue
        yield class_id, lx1, lx2, ly1, ly2


def count_objects_in_crop(records: List[DetectionRecord], crop_rect: Tuple[int, int, int, int]) -> int:
    return sum(1 for _ in iter_crop_records_for_export(records, crop_rect))


class MainWindow(QMainWindow):
    def __init__(
        self,
        in_orig: Optional[str],
        in_yolo: str,
        in_csv: Optional[str],
        out_dir_images: Path,
        out_dir_labels: Path,
        records: List[DetectionRecord],
        preview_qimage: QImage,
        orig_size: Tuple[int, int],
        preview_mode: bool = False,
    ):
        super().__init__()
        self.in_orig = in_orig
        self.in_yolo = in_yolo
        self.in_csv = in_csv
        self.preview_mode = preview_mode
        self.out_dir_images = out_dir_images
        self.out_dir_labels = out_dir_labels
        self.records = records
        self.is_edit_mode = False
        self._save_in_progress = False
        self._initialized_view = False
        self._detail_worker_total = 1
        self._detail_worker_done = 0
        self._detail_step = 0
        self._detail_total = 1
        self._detail_percent_display = 0
        self._detail_tick = 0

        self.setWindowTitle("YOLO Label Reassembler")
        self.resize(1500, 900)

        pixmap = QPixmap.fromImage(preview_qimage)
        self.canvas = ImageCanvas(pixmap, in_yolo, orig_size, self)

        right = QWidget(self)
        right.setFixedWidth(430)
        right_layout = QVBoxLayout(right)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(QLabel(f"Original image: {Path(in_orig).name if in_orig else '(preview only)'}"))
        right_layout.addWidget(QLabel(f"Preview image: {Path(in_yolo).name}"))
        right_layout.addWidget(QLabel(f"CSV file: {Path(in_csv).name if in_csv else '(preview only)'}"))
        right_layout.addWidget(QLabel(f"Output image dir: {self.out_dir_images}"))
        right_layout.addWidget(QLabel(f"Output label dir: {self.out_dir_labels}"))
        right_layout.addSpacing(8)

        self.btn_edit = QPushButton("Edit")
        self.btn_save = QPushButton("Save")
        self.btn_edit.clicked.connect(self.on_edit_clicked)
        self.btn_save.clicked.connect(self.on_save_clicked)
        right_layout.addWidget(self.btn_edit)
        right_layout.addWidget(self.btn_save)
        if self.preview_mode:
            self.btn_edit.setEnabled(False)
            self.btn_save.setEnabled(False)

        self.save_progress = QProgressBar()
        self.save_progress.setRange(0, 100)
        self.save_progress.setValue(0)
        self.save_progress.setFormat("Saving... %p%")
        self.save_progress.setVisible(False)
        right_layout.addWidget(self.save_progress)

        self.msg_label = QLabel(
            "Preview mode: edit/save disabled." if self.preview_mode else "Edit mode disabled."
        )
        self.msg_label.setWordWrap(True)
        right_layout.addWidget(self.msg_label)

        right_layout.addWidget(QLabel("Saved files:"))
        self.saved_list = QListWidget()
        right_layout.addWidget(self.saved_list)

        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 100)
        self.render_progress.setValue(0)
        self.render_progress.setFormat("Detail loading: idle")
        self.render_progress.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        right_layout.addWidget(self.render_progress)
        self.view_info_label = QLabel("")
        self.view_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        right_layout.addWidget(self.view_info_label)

        self._render_blink_timer = QTimer(self)
        self._render_blink_timer.setInterval(120)
        self._render_blink_timer.timeout.connect(self._blink_render_progress)
        self._render_blink_phase = 0

        central = QWidget(self)
        hl = QHBoxLayout(central)
        hl.addWidget(self.canvas, 3)
        hl.addWidget(right, 1)
        self.setCentralWidget(central)
        self.canvas.view_metrics_changed.connect(self._on_view_metrics_changed)
        self.canvas.detail_loading_changed.connect(self._on_detail_loading_changed)
        self.canvas.detail_progress_changed.connect(self._on_detail_progress_changed)
        self.canvas.draft_rect_changed.connect(self._on_draft_rect_changed)
        self._selection_count_timer = QTimer(self)
        self._selection_count_timer.setSingleShot(True)
        self._selection_count_timer.setInterval(80)
        self._selection_count_timer.timeout.connect(self._update_edit_mode_message)

    def set_edit_mode(self, enabled: bool) -> None:
        self.is_edit_mode = enabled
        self.canvas.set_edit_mode(enabled)
        if enabled:
            self.btn_edit.setText("Exit Edit")
            self._update_edit_mode_message()
        else:
            self._selection_count_timer.stop()
            self.btn_edit.setText("Edit")
            self.msg_label.setText("Edit mode disabled.")

    def on_edit_clicked(self) -> None:
        if self.preview_mode:
            return
        self.set_edit_mode(not self.is_edit_mode)

    def on_save_clicked(self) -> None:
        if self.preview_mode:
            QMessageBox.warning(self, "Save", "Save is disabled in preview mode.")
            return
        if self.in_orig is None:
            QMessageBox.warning(self, "Save", "Original image path is not set.")
            return
        if not self.is_edit_mode:
            QMessageBox.warning(self, "Save", "Save is available only in Edit mode.")
            return

        rect = self.canvas.get_draft_rect_in_orig()
        if rect is None:
            QMessageBox.warning(self, "Save", "No valid red rectangle to save.")
            return

        out_png, out_txt = unique_output_paths(self.out_dir_images, self.out_dir_labels, self.in_orig)

        self.btn_edit.setEnabled(False)
        self.btn_save.setEnabled(False)
        self._save_in_progress = True
        self.canvas.set_edit_interaction_locked(True)
        self._set_render_loading(False)
        self.save_progress.setValue(0)
        self.save_progress.setVisible(True)
        self.msg_label.setText("Saving...")

        self._save_thread = QThread(self)
        self._save_worker = SaveWorker(self.in_orig, out_png, out_txt, self.records, rect)
        self._save_worker.moveToThread(self._save_thread)

        self._save_thread.started.connect(self._save_worker.run)
        self._save_worker.progress.connect(self._on_save_progress)
        self._save_worker.done.connect(self._on_save_done)
        self._save_worker.failed.connect(self._on_save_failed)
        self._save_worker.done.connect(self._save_thread.quit)
        self._save_worker.failed.connect(self._save_thread.quit)
        self._save_thread.finished.connect(self._save_worker.deleteLater)
        self._save_thread.finished.connect(self._save_thread.deleteLater)
        self._save_thread.start()

    def _on_save_progress(self, value: int, message: str) -> None:
        self.save_progress.setValue(value)
        self.msg_label.setText(message)

    def _restore_after_save(self) -> None:
        self.btn_edit.setEnabled(not self.preview_mode)
        self.btn_save.setEnabled(not self.preview_mode)
        self._save_in_progress = False
        self.canvas.set_edit_interaction_locked(False)
        self.save_progress.setVisible(False)

    @Slot()
    def _on_draft_rect_changed(self) -> None:
        if not self.is_edit_mode or self._save_in_progress:
            return
        self._selection_count_timer.start()

    def _update_edit_mode_message(self) -> None:
        if not self.is_edit_mode:
            return
        rect = self.canvas.get_draft_rect_in_orig()
        if rect is None:
            self.msg_label.setText("Edit mode: draw one red rectangle on the image.")
            return
        x1, y1, x2, y2 = rect
        count = count_objects_in_crop(self.records, rect)
        self.msg_label.setText(
            f"Edit mode: selected {count} objects in rectangle ({x2 - x1} x {y2 - y1} px)."
        )

    def _on_save_done(self, out_png: str, out_txt: str, n: int) -> None:
        self.canvas.finalize_current_rect()
        self.set_edit_mode(False)
        self.saved_list.addItem(QListWidgetItem(Path(out_png).name))
        self.saved_list.addItem(QListWidgetItem(Path(out_txt).name))
        self.msg_label.setText(f"Saved: {Path(out_png).name}, {Path(out_txt).name} ({n} labels)")
        self._restore_after_save()

    def _on_save_failed(self, err: str) -> None:
        QMessageBox.critical(self, "Save error", err)
        self.msg_label.setText("Save failed.")
        self._restore_after_save()

    def _blink_render_progress(self) -> None:
        # Fill in-between worker step updates so users can see continuous progress.
        self._detail_tick += 1
        if self._detail_total <= 0:
            self._detail_total = 1
        lower = int((self._detail_step - 1) * 100 / self._detail_total) if self._detail_step > 0 else 0
        upper = int(self._detail_step * 100 / self._detail_total) if self._detail_step > 0 else 10
        upper = min(99, max(lower + 1, upper))
        if self._detail_percent_display < upper:
            self._detail_percent_display += 1
        self.render_progress.setValue(self._detail_percent_display)
        self.render_progress.setFormat(
            f"Detail loading: {self._detail_percent_display}% "
            f"(step {self._detail_step}/{self._detail_total}, tick {self._detail_tick})"
        )

    def _set_render_loading(self, loading: bool) -> None:
        if loading and not self._save_in_progress:
            self._detail_worker_done = 0
            self._detail_step = 0
            self._detail_total = 1
            self._detail_percent_display = 1
            self._detail_tick = 0
            self.render_progress.setValue(self._detail_percent_display)
            self.render_progress.setFormat("Detail loading: 1% (starting...)")
            if not self._render_blink_timer.isActive():
                self._render_blink_timer.start()
        else:
            self._render_blink_timer.stop()
            self._detail_worker_done = self._detail_worker_total
            self._detail_percent_display = 100
            self.render_progress.setValue(100)
            self.render_progress.setFormat(
                f"Detail loading: done ({self._detail_worker_done}/{self._detail_worker_total})"
            )

    def _on_detail_loading_changed(self, loading: bool) -> None:
        self._set_render_loading(loading)

    def _on_detail_progress_changed(self, step: int, total: int, message: str) -> None:
        if total <= 0:
            return
        self._detail_step = max(1, step)
        self._detail_total = max(1, total)
        pct = max(1, min(99, int(round(self._detail_step * 100.0 / self._detail_total))))
        self._detail_percent_display = max(self._detail_percent_display, pct)
        self.render_progress.setValue(self._detail_percent_display)
        self.render_progress.setFormat(
            f"Detail loading: {self._detail_percent_display}% "
            f"(step {self._detail_step}/{self._detail_total}, {message})"
        )

    def _on_view_metrics_changed(self, text: str) -> None:
        self.view_info_label.setText(text)

    def closeEvent(self, event) -> None:
        self.canvas.shutdown()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._initialized_view:
            self._initialized_view = True
            QTimer.singleShot(0, self.canvas.initialize_view)


def validate_input_paths(args: argparse.Namespace) -> None:
    has_in_orig = bool(args.in_orig)
    has_in_csv = bool(args.in_csv)
    if has_in_orig != has_in_csv:
        print(
            "Warning: --in_orig and --in_csv must be specified together. "
            "Provide both options or neither (preview mode).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    for key in ("in_yolo", "in_orig", "in_csv"):
        val = getattr(args, key)
        if not val:
            continue
        p = Path(val)
        if not p.is_file():
            print(f"Error: file not found: {p}", file=sys.stderr)
            raise SystemExit(2)


def main() -> None:
    startup = StartupProgress(total_units=20)
    startup.update("Parsing arguments...", units=1.0)
    args = parse_args()
    startup.update("Validating input files...", units=1.0)
    validate_input_paths(args)
    startup.update("Checking pyvips/libvips...", units=1.0)
    require_pyvips_and_libvips()
    startup.update("Resolving output directories...", units=1.0)
    out_dir_images = resolve_output_dir(args.out_dir_images, "image output")
    out_dir_labels = resolve_output_dir(args.out_dir_labels, "label output")
    preview_mode = not args.in_orig and not args.in_csv

    records: List[DetectionRecord] = []
    if preview_mode:
        startup.update("Skipping CSV load in preview mode...", units=2.0)
    else:
        startup.update("Loading CSV detections...", units=2.0)
        records = load_detection_csv(args.in_csv)

    startup.update("Reading input image metadata...", units=2.0)
    yolo_size = read_image_size(args.in_yolo)
    orig_size = yolo_size
    if preview_mode:
        startup.update("Preview mode: using --in_yolo size as base...", units=0.8)
    else:
        orig_size = read_image_size(args.in_orig)
        if yolo_size != orig_size:
            print(
                f"Warning: image size mismatch. in_orig={orig_size}, in_yolo={yolo_size}. "
                "Both images must have exactly the same width/height.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    startup.update("Preparing preview image...", units=1.0)

    def _preview_progress(msg: str, advance: bool) -> None:
        if advance:
            startup.update(msg, units=0.8)
        else:
            startup.pulse(msg)

    qimg, preview_orig_w, preview_orig_h = load_preview_qimage(args.in_yolo, progress_cb=_preview_progress)
    startup.update("Preparing preview image... done", units=6.0)
    if (preview_orig_w, preview_orig_h) != yolo_size:
        print(
            f"Warning: preview loader size mismatch: header={yolo_size}, loader={(preview_orig_w, preview_orig_h)}",
            file=sys.stderr,
        )

    startup.update("Starting GUI...", units=2.0)
    app = QApplication(sys.argv)
    w = MainWindow(
        in_orig=args.in_orig,
        in_yolo=args.in_yolo,
        in_csv=args.in_csv,
        out_dir_images=out_dir_images,
        out_dir_labels=out_dir_labels,
        records=records,
        preview_qimage=qimg,
        orig_size=orig_size,
        preview_mode=preview_mode,
    )
    startup.update("Showing window...", units=3.0)
    w.show()
    startup.finish("Ready.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
