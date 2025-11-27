from PySide6.QtWidgets import QGraphicsView, QGraphicsEllipseItem, QGraphicsView
from PySide6.QtGui import QWheelEvent, QMouseEvent, QBrush
from PySide6.QtCore import Qt



class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.zoom_factor = 1.15
        self._zoom = 0
        self._max_zoom = 10
        self._min_zoom = -10

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0 and self._zoom < self._max_zoom:
            zoom = self.zoom_factor
            self._zoom += 1
        elif event.angleDelta().y() < 0 and self._zoom > self._min_zoom:
            zoom = 1 / self.zoom_factor
            self._zoom -= 1
        else:
            return

        self.scale(zoom, zoom)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # Reset zoom on double click
        self.resetTransform()
        self._zoom = 0
        self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)

    def export_view(self, filename: str):
        from PySide6.QtGui import QImage, QPainter

        rect = self.viewport().rect()
        image = QImage(rect.width(), rect.height(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)

        painter = QPainter(image)
        self.render(painter, target=image.rect(), source=rect)
        painter.end()

        image.save(filename)


    def get_scene_position(self, pos):
        """Convert viewport position to scene position."""
        return self.mapToScene(pos)


    def draw_annotation_point(self, scene_pos, color=Qt.GlobalColor.red, radius=4):
        """Draw a circle at a scene position."""
        ellipse = QGraphicsEllipseItem(
            scene_pos.x() - radius, scene_pos.y() - radius, radius * 2, radius * 2
        )
        ellipse.setBrush(QBrush(color))
        ellipse.setPen(Qt.PenStyle.NoPen)
        self.scene().addItem(ellipse)
        return ellipse


