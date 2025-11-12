from turtle import color
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QGraphicsSceneMouseEvent,
    QGraphicsItemGroup,
    QGraphicsItem,
    QGraphicsSimpleTextItem
)
from PySide6.QtGui import (
    QPen,
    QColor,
    QPainterPath,
    QBrush
)
from PySide6.QtCore import (
    QObject,
    Qt,
    QRectF,
    QPointF,
    Signal
)
import sys
from typing import Optional


HANDLE_SIZE = 10
HANDLE_TYPES = [
    "top_left", "top_center", "top_right",
    "right_center", "bottom_right", "bottom_center",
    "bottom_left", "left_center", "center_move"
]


class HandleItem(QObject, QGraphicsRectItem):
    hovered = Signal(bool)  # True = mouse over, False = mouse leave

    def __init__(self, name: str, parent: "ResizableRectItem"):
        QObject.__init__(self)
        QGraphicsRectItem.__init__(self, 0, 0, HANDLE_SIZE, HANDLE_SIZE, parent)

        self.name = name
        self.parentItemRef = parent
        self.setBrush(QBrush(QColor("blue")))
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setZValue(1)


    def hoverEnterEvent(self, event):
        self.hovered.emit(True)
        self.setBrush(QBrush(QColor("orange")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered.emit(False)
        self.setBrush(QBrush(QColor("blue")))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.setBrush(QBrush(QColor("red")))

        if self.name == "center_move":
            self._start_pos = event.scenePos()
        else:
            # Posición en coordenadas del padre (locales)
            self._start_pos = self.parentItemRef.mapFromScene(event.scenePos())

        event.accept()

    def mouseMoveEvent(self, event):
        if self.name == "center_move":
            # Movimiento global
            delta = event.scenePos() - self._start_pos
            self._start_pos = event.scenePos()
            self.parentItemRef.moveBy(delta.x(), delta.y())
        else:
            # Redimensionamiento local
            current_pos = self.parentItemRef.mapFromScene(event.scenePos())
            delta = current_pos - self._start_pos
            self._start_pos = current_pos
            self.parentItemRef._resize_by_handle(self.name, delta)

        event.accept()

    def mouseReleaseEvent(self, event):
        self.setBrush(QBrush(QColor("blue")))
        event.accept()


class ResizableRectItem(QObject, QGraphicsItem):
    hover_state_changed = Signal(bool)
    mouse_hover_state = Signal(bool)

    geometry_committed = Signal(object)
    deleted = Signal()

    def __init__(
        self,
        rect: QRectF,
        draw_ellipse: bool = False,
        color: QColor = QColor("red"),
        parent: QGraphicsItem | None = None
    ):
        QObject.__init__(self)
        QGraphicsItem.__init__(self, parent)

        self.rect = rect
        self.pen_color = color
        self.draw_ellipse = draw_ellipse
        self.handles: dict[str, HandleItem] = {}
        self.label_item: Optional[QGraphicsSimpleTextItem] = None

        for name in HANDLE_TYPES:
            handle = HandleItem(name, self)
            handle.hovered.connect(self.hover_state_changed.emit)
            self.handles[name] = handle

        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.AllButtons)
        self.show_handles(False)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(0)

        self._update_handle_positions()


    def set_pen_color(self, color: QColor):
        self.pen_color = color
        self.update()


    def setRect(self, new_rect: QRectF):
        self.prepareGeometryChange()
        self.rect = new_rect
        self._update_handle_positions()
        self.update_label_position()
        self.update()


    def show_handles(self, show: bool):
        for handle in self.handles.values():
            handle.setVisible(show)


    def hoverEnterEvent(self, event):
        self.show_handles(True)
        self.hover_state_changed.emit(True)
        self.mouse_hover_state.emit(True)
        super().hoverEnterEvent(event)


    def hoverLeaveEvent(self, event):
        self.show_handles(False)
        self.hover_state_changed.emit(False)
        self.mouse_hover_state.emit(False)
        super().hoverLeaveEvent(event)


    def update_label_position(self):
        if self.label_item:
            corner = self.mapToScene(self.rect.bottomLeft())
            self.label_item.setPos(corner + QPointF(0, 5))


    def boundingRect(self) -> QRectF:
        margin = max(HANDLE_SIZE, 4.0)
        return self.rect.adjusted(-margin, -margin, margin, margin)


    def moveBy(self, dx: float, dy: float) -> None:
        delta_scene = QPointF(dx, dy)
        start_scene_pos = self.scenePos()
        new_scene_pos = start_scene_pos + delta_scene
        new_local_pos = self.mapFromScene(new_scene_pos)
        self.setPos(self.pos() + new_local_pos - self.mapFromScene(start_scene_pos))

        if self.label_item:
            self.update_label_position()

        # Emitir cambio de geometría
        self.geometry_committed.emit(self.rect)


    def paint(self, painter, option, widget=None):
        pen = QPen(self.pen_color, 2)
        painter.setPen(pen)

        if self.draw_ellipse:
            painter.drawEllipse(self.rect)
        else:
            painter.drawRect(self.rect)

    def _update_handle_positions(self):
        r = self.rect
        size = HANDLE_SIZE / 2

        positions = {
            "top_left": QPointF(r.left() - size, r.top() - size),
            "top_center": QPointF(r.center().x() - size, r.top() - size),
            "top_right": QPointF(r.right() - size, r.top() - size),
            "right_center": QPointF(r.right() - size, r.center().y() - size),
            "bottom_right": QPointF(r.right() - size, r.bottom() - size),
            "bottom_center": QPointF(r.center().x() - size, r.bottom() - size),
            "bottom_left": QPointF(r.left() - size, r.bottom() - size),
            "left_center": QPointF(r.left() - size, r.center().y() - size),
            "center_move": QPointF(r.center().x() - size, r.center().y() - size)
        }

        for name, handle in self.handles.items():
            pos = positions[name]
            handle.setRect(pos.x(), pos.y(), HANDLE_SIZE, HANDLE_SIZE)

        self.geometry_committed.emit(self.rect)


    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._update_handle_positions()
            self.update_label_position()
        return super().itemChange(change, value)


    def contains_point(self, scene_point: QPointF) -> bool:
        return self.shape().contains(self.mapFromScene(scene_point))

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect)  # Usa el rect local
        return path

    def _resize_by_handle(self, handle_name: str, delta: QPointF):
        r = QRectF(self.rect)

        if handle_name   == "top_left"     : r.setTopLeft(r.topLeft() + delta)
        elif handle_name == "top_center"   : r.setTop(r.top() + delta.y())
        elif handle_name == "top_right"    : r.setTopRight(r.topRight() + delta)
        elif handle_name == "right_center" : r.setRight(r.right() + delta.x())
        elif handle_name == "bottom_right" : r.setBottomRight(r.bottomRight() + delta)
        elif handle_name == "bottom_center": r.setBottom(r.bottom() + delta.y())
        elif handle_name == "bottom_left"  : r.setBottomLeft(r.bottomLeft() + delta)
        elif handle_name == "left_center"  : r.setLeft(r.left() + delta.x())

        self.setRect(r.normalized())



# --- Métodos esperados por AnnotationItem ---
    def get_pen_color(self) -> QColor:
        return self.pen_color  # solo lectura, sin modificar

    def get_label_text(self) -> str:
        return "" if not self.label_item else self.label_item.text()

    def set_label_item(self, label_item: QGraphicsSimpleTextItem):
        self.label_item = label_item
        self.update_label_position()

    def setRotation(self, angle: float):
        super().setRotation(angle)
        self.geometry_committed.emit(self.rect)


    # ------------- Eventos de Mouse ---------------
    # def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
    #     pass

    # def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
    #     pass

    # def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
    #     pass




# -----------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    view.setRenderHint(view.renderHints())
    view.setWindowTitle("Resizable Item Demo")
    view.resize(800, 600)

    rect = QRectF(100, 100, 200, 150)
    item = ResizableRectItem(rect, draw_ellipse=False)  # Cambia a True para elipse
    scene.addItem(item)

    view.show()
    sys.exit(app.exec())
