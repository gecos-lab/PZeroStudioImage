"""Interactive, resolution-preserving image canvas used by the desktop shell."""

from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .theme import COLORS


class InteractionMode(str, Enum):
    """Pointer behaviours supported by :class:`ImageCanvas`."""

    PAN = "pan"
    TRACE = "trace"
    INSPECT = "inspect"


ImageSource = Union[str, Path, QImage, QPixmap]
PointLike = Union[QPointF, QPoint, Tuple[float, float], Sequence[float]]


def _as_pixmap(source: ImageSource) -> QPixmap:
    if isinstance(source, QPixmap):
        return QPixmap(source)
    if isinstance(source, QImage):
        return QPixmap.fromImage(source)
    return QPixmap(str(source))


def _as_point(value: PointLike) -> QPointF:
    if isinstance(value, QPointF):
        return QPointF(value)
    if isinstance(value, QPoint):
        return QPointF(value)
    return QPointF(float(value[0]), float(value[1]))


class ImageCanvas(QGraphicsView):
    """A graphics-view canvas with pan, cursor-centred zoom, and trace anchors.

    The canvas never changes the source image resolution.  All overlay geometry is
    expressed in source-pixel coordinates, which keeps it suitable for a future
    geospatial controller.
    """

    startPointChanged = pyqtSignal(QPointF)
    endPointChanged = pyqtSignal(QPointF)
    traceRequested = pyqtSignal(QPointF, QPointF)
    cursorPositionChanged = pyqtSignal(QPointF)
    cursorLeftImage = pyqtSignal()
    zoomChanged = pyqtSignal(float)
    modeChanged = pyqtSignal(str)

    MIN_SCALE = 0.035
    MAX_SCALE = 64.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ImageCanvas")
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._image_item = QGraphicsPixmapItem()
        self._image_item.setTransformationMode(Qt.SmoothTransformation)
        self._image_item.setZValue(0)
        self._scene.addItem(self._image_item)

        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setTransformationMode(Qt.SmoothTransformation)
        self._overlay_item.setOpacity(0.55)
        self._overlay_item.setZValue(5)
        self._scene.addItem(self._overlay_item)

        self._trace_items = []
        self._start_marker: Optional[QGraphicsEllipseItem] = None
        self._end_marker: Optional[QGraphicsEllipseItem] = None
        self._guide_item: Optional[QGraphicsPathItem] = None
        self._start_point: Optional[QPointF] = None
        self._end_point: Optional[QPointF] = None
        self._mode = InteractionMode.PAN
        self._anchor_input_enabled = True
        self._fit_scale = 1.0
        self._middle_panning = False
        self._last_pan_position = QPoint()

        self.setBackgroundBrush(QColor(COLORS["canvas"]))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setRenderHints(
            QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
            | QPainter.TextAntialiasing
        )
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.StrongFocus)

    @property
    def has_image(self) -> bool:
        return not self._image_item.pixmap().isNull()

    @property
    def interaction_mode(self) -> InteractionMode:
        return self._mode

    @property
    def start_point(self) -> Optional[QPointF]:
        return QPointF(self._start_point) if self._start_point is not None else None

    @property
    def end_point(self) -> Optional[QPointF]:
        return QPointF(self._end_point) if self._end_point is not None else None

    def image_size(self):
        return self._image_item.pixmap().size()

    def set_image(self, source: ImageSource, fit: bool = True) -> bool:
        """Display an image without resampling it; return ``False`` on load error."""

        pixmap = _as_pixmap(source)
        if pixmap.isNull():
            return False
        self._image_item.setPixmap(pixmap)
        bounds = QRectF(pixmap.rect())
        self._scene.setSceneRect(bounds)
        self.clear_overlay()
        self.clear_traces()
        self.clear_anchors()
        if fit:
            self.fit_to_image()
        self.viewport().update()
        return True

    def clear_image(self) -> None:
        self._image_item.setPixmap(QPixmap())
        self.clear_overlay()
        self.clear_traces()
        self.clear_anchors()
        self.resetTransform()
        self._scene.setSceneRect(QRectF())
        self.viewport().update()

    def set_probability_overlay(self, source: ImageSource, opacity: float = 0.55) -> bool:
        pixmap = _as_pixmap(source)
        if pixmap.isNull() or not self.has_image:
            return False
        if pixmap.size() != self._image_item.pixmap().size():
            pixmap = pixmap.scaled(
                self._image_item.pixmap().size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
        self._overlay_item.setPixmap(pixmap)
        self.set_overlay_opacity(opacity)
        self._overlay_item.setVisible(True)
        return True

    def clear_overlay(self) -> None:
        self._overlay_item.setPixmap(QPixmap())

    def set_overlay_opacity(self, opacity: float) -> None:
        self._overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))

    def set_overlay_visible(self, visible: bool) -> None:
        self._overlay_item.setVisible(bool(visible))

    def set_traces_visible(self, visible: bool) -> None:
        for item in self._trace_items:
            item.setVisible(bool(visible))

    def clear_traces(self) -> None:
        for item in self._trace_items:
            self._scene.removeItem(item)
        self._trace_items.clear()

    def set_traces(self, traces: Iterable[Iterable[PointLike]]) -> None:
        """Replace vector traces using source-image coordinate sequences."""

        self.clear_traces()
        for points in traces:
            polygon = QPolygonF([_as_point(point) for point in points])
            if polygon.isEmpty():
                continue
            path = QPainterPath(polygon.first())
            for point in polygon[1:]:
                path.lineTo(point)
            item = QGraphicsPathItem(path)
            pen = QPen(
                QColor(COLORS["blue"]),
                2.2,
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
            item.setZValue(20)
            self._scene.addItem(item)
            self._trace_items.append(item)

    def append_trace(self, points: Iterable[PointLike], selected: bool = False) -> None:
        converted = [_as_point(point) for point in points]
        if not converted:
            return
        path = QPainterPath(converted[0])
        for point in converted[1:]:
            path.lineTo(point)
        item = QGraphicsPathItem(path)
        colour = COLORS["warning"] if selected else COLORS["blue"]
        width = 3.0 if selected else 2.2
        pen = QPen(QColor(colour), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(20)
        self._scene.addItem(item)
        self._trace_items.append(item)

    def set_interaction_mode(self, mode: Union[InteractionMode, str]) -> None:
        mode = InteractionMode(mode)
        if mode == self._mode:
            return
        self._mode = mode
        self.setDragMode(
            QGraphicsView.ScrollHandDrag
            if mode == InteractionMode.PAN
            else QGraphicsView.NoDrag
        )
        self.setCursor(Qt.ArrowCursor if mode != InteractionMode.PAN else Qt.OpenHandCursor)
        self.modeChanged.emit(mode.value)

    def set_anchor_input_enabled(self, enabled: bool) -> None:
        """Allow or block creation of trace anchors without disabling navigation."""

        self._anchor_input_enabled = bool(enabled)

    def fit_to_image(self) -> None:
        if not self.has_image:
            return
        self.resetTransform()
        self.fitInView(self._image_item.boundingRect(), Qt.KeepAspectRatio)
        self._fit_scale = self.transform().m11()
        self._emit_zoom()

    def actual_pixels(self) -> None:
        if not self.has_image:
            return
        self.resetTransform()
        self._emit_zoom()

    def zoom_in(self) -> None:
        self._apply_zoom(1.25)

    def zoom_out(self) -> None:
        self._apply_zoom(0.8)

    def clear_anchors(self) -> None:
        for marker in (self._start_marker, self._end_marker, self._guide_item):
            if marker is not None and marker.scene() is self._scene:
                self._scene.removeItem(marker)
        self._start_marker = None
        self._end_marker = None
        self._guide_item = None
        self._start_point = None
        self._end_point = None
        self.viewport().update()

    def set_anchors(
        self, start: Optional[PointLike], end: Optional[PointLike] = None
    ) -> None:
        self.clear_anchors()
        if start is not None:
            self._set_start(_as_point(start))
        if end is not None:
            self._set_end(_as_point(end))

    def _image_contains(self, scene_position: QPointF) -> bool:
        if not self.has_image:
            return False
        size = self.image_size()
        return (
            0.0 <= scene_position.x() < float(size.width())
            and 0.0 <= scene_position.y() < float(size.height())
        )

    def _nearest_pixel(self, scene_position: QPointF) -> QPointF:
        """Snap a valid scene coordinate to an in-bounds integer pixel."""

        size = self.image_size()
        x = min(size.width() - 1, max(0, round(scene_position.x())))
        y = min(size.height() - 1, max(0, round(scene_position.y())))
        return QPointF(float(x), float(y))

    def _new_marker(self, point: QPointF, colour: str) -> QGraphicsEllipseItem:
        radius = 5.0
        marker = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
        marker.setPos(point)
        marker.setBrush(QColor(colour))
        marker.setPen(QPen(QColor("#FFFFFF"), 1.5))
        marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        marker.setZValue(30)
        self._scene.addItem(marker)
        return marker

    def _set_start(self, point: QPointF) -> None:
        self.clear_anchors()
        self._start_point = QPointF(point)
        self._start_marker = self._new_marker(point, COLORS["primary"])
        self.startPointChanged.emit(QPointF(point))

    def _set_end(self, point: QPointF) -> None:
        self._end_point = QPointF(point)
        if self._end_marker is not None:
            self._scene.removeItem(self._end_marker)
        self._end_marker = self._new_marker(point, COLORS["warning"])
        self._update_guide(point)
        self.endPointChanged.emit(QPointF(point))

    def _update_guide(self, end: QPointF) -> None:
        if self._start_point is None:
            return
        path = QPainterPath(self._start_point)
        path.lineTo(end)
        if self._guide_item is None:
            self._guide_item = QGraphicsPathItem()
            self._guide_item.setZValue(18)
            self._scene.addItem(self._guide_item)
        self._guide_item.setPath(path)
        self._guide_item.setPen(QPen(QColor("#DCE5EC"), 1.0, Qt.DashLine))

    def _apply_zoom(self, factor: float) -> None:
        if not self.has_image:
            return
        current = self.transform().m11()
        target = current * factor
        if factor > 1.0:
            if current >= self.MAX_SCALE:
                return
            factor = min(factor, self.MAX_SCALE / current)
        else:
            if current <= self.MIN_SCALE:
                return
            factor = max(factor, self.MIN_SCALE / current)
        self.scale(factor, factor)
        self._emit_zoom()

    def _emit_zoom(self) -> None:
        self.zoomChanged.emit(self.transform().m11() * 100.0)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.has_image:
            event.ignore()
            return
        self._apply_zoom(1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._middle_panning = True
            self._last_pan_position = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if (
            event.button() == Qt.LeftButton
            and self._mode == InteractionMode.TRACE
            and self.has_image
            and self._anchor_input_enabled
        ):
            point = self.mapToScene(event.pos())
            if self._image_contains(point):
                point = self._nearest_pixel(point)
                if self._start_point is None or self._end_point is not None:
                    self._set_start(point)
                else:
                    self._set_end(point)
                    self.traceRequested.emit(QPointF(self._start_point), QPointF(point))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._middle_panning:
            delta = event.pos() - self._last_pan_position
            self._last_pan_position = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        point = self.mapToScene(event.pos())
        if self._image_contains(point):
            self.cursorPositionChanged.emit(point)
            if (
                self._mode == InteractionMode.TRACE
                and self._anchor_input_enabled
                and self._start_point is not None
                and self._end_point is None
            ):
                self._update_guide(point)
        else:
            self.cursorLeftImage.emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton and self._middle_panning:
            self._middle_panning = False
            self.setCursor(
                Qt.OpenHandCursor if self._mode == InteractionMode.PAN else Qt.ArrowCursor
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.has_image:
            return
        # Keep the initial fitted view useful while allowing explicit user zoom.
        if abs(self.transform().m11() - self._fit_scale) < 1e-6:
            self.fit_to_image()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if self.has_image:
            return

        painter.save()
        viewport_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        center = viewport_rect.center()
        painter.setPen(QColor(COLORS["text"]))
        title_font = painter.font()
        title_font.setPointSize(14)
        title_font.setWeight(QFont.DemiBold)
        painter.setFont(title_font)
        title_rect = QRectF(center.x() - 210, center.y() - 34, 420, 30)
        painter.drawText(title_rect, Qt.AlignCenter, "Open a geological image")

        painter.setPen(QColor(COLORS["text_muted"]))
        body_font = painter.font()
        body_font.setPointSize(10)
        body_font.setWeight(QFont.Normal)
        painter.setFont(body_font)
        body_rect = QRectF(center.x() - 260, center.y() + 2, 520, 24)
        painter.drawText(
            body_rect,
            Qt.AlignCenter,
            "Retains full resolution and spatial coordinates",
        )
        painter.restore()
