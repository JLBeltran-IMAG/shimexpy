"""Alternative scatter plot implementation without OpenGL dependencies."""
from typing import Dict, Optional, List
import numpy as np

from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar, QFileDialog,
    QGraphicsScene, QGraphicsView
)
from PySide6.QtGui import (
    QPainter, QColor, QPixmap, QAction
)

from ..logic.annotation_item import AnnotationItem
from .scatter_compare_qt import _pair_pixels

class ScatterView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        # No antialiasing o viewport update mode para evitar problemas de OpenGL
        self._points = {}  # id -> [(x,y), ...]
        self._colors = {}  # id -> QColor
        self._names = {}   # id -> str
        
    def clear_series(self, series_id: str):
        if series_id in self._points:
            del self._points[series_id]
            del self._colors[series_id]
            del self._names[series_id]
            self.scene().update()
            
    def update_series(self, series_id: str, points, color: QColor, name: str):
        self._points[series_id] = points
        self._colors[series_id] = color
        self._names[series_id] = name
        self.scene().update()
        
    def drawBackground(self, painter: QPainter, rect):
        super().drawBackground(painter, rect)
        painter.fillRect(rect, QColor(255, 255, 255))  # white
        
    def drawForeground(self, painter: QPainter, rect):
        super().drawForeground(painter, rect)
        # Dibujar puntos
        for sid, points in self._points.items():
            color = self._colors[sid]
            painter.setPen(color)
            painter.setBrush(color)
            points_array = points if isinstance(points, list) else points.tolist()
            for x, y in points_array:
                painter.drawEllipse(QPointF(x*100, y*100), 2.0, 2.0)

class ScatterCompareSimple(QMainWindow):
    """
    Ventana de scatter sin OpenGL, escuchando dos ImageContainer2D.
    Versión simplificada que usa QGraphicsView en lugar de QChartView.
    """
    
    def __init__(self, container_a, container_b, max_points: int = 200_000, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Scatter Compare (Simple)")
        
        self.a = container_a
        self.b = container_b
        self.max_points = int(max_points)
        
        # ---- UI ----
        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)
        
        self.view = ScatterView()
        self.view.setMinimumSize(700, 500)
        self.view.setMaximumHeight(2000)
        lay.addWidget(self.view)
        
        # Toolbar
        tb = QToolBar(self)
        act_save = QAction("Guardar PNG", self)
        act_save.triggered.connect(self._save_png)
        tb.addAction(act_save)
        self.addToolBar(tb)
        
        # series por id (puede ser lista o numpy array)
        self._series_by_id = {}
        
        # Escuchar eventos de anotaciones
        self._wire_to_managers()
        
        # Primer dibujado
        self.refresh_all()
    
    def _chain(self, prev, mine):
        def inner(ann: AnnotationItem):
            if self.a.ann_mgr.get(ann.id) is ann:
                if ann.pixel_values is None or ann.pixel_values.size == 0:
                    ann.capture_pixels(self.a)
            if self.b.ann_mgr.get(ann.id) is ann:
                if ann.pixel_values is None or ann.pixel_values.size == 0:
                    ann.capture_pixels(self.b)
            if prev:
                prev(ann)
            mine(ann)
        return inner
    
    def _wire_to_managers(self):
        # A
        pa = self.a.ann_mgr.on_added
        pu = self.a.ann_mgr.on_updated
        pr = getattr(self.a.ann_mgr, "on_removed", None)
        self.a.ann_mgr.on_added = self._chain(pa, lambda ann: self._update_one(ann.id))
        self.a.ann_mgr.on_updated = self._chain(pu, lambda ann: self._update_one(ann.id))
        if pr is not None:
            self.a.ann_mgr.on_removed = self._chain(pr, lambda ann: self._remove_one(ann.id))
        
        # B
        pb = self.b.ann_mgr.on_added
        pu2 = self.b.ann_mgr.on_updated
        pr2 = getattr(self.b.ann_mgr, "on_removed", None)
        self.b.ann_mgr.on_added = self._chain(pb, lambda ann: self._update_one(ann.id))
        self.b.ann_mgr.on_updated = self._chain(pu2, lambda ann: self._update_one(ann.id))
        if pr2 is not None:
            self.b.ann_mgr.on_removed = self._chain(pr2, lambda ann: self._remove_one(ann.id))
    
    def refresh_all(self):
        """Recalcula todas las series para ids comunes en A y B."""
        ids_a = {ann.id for ann in self.a.ann_mgr.items()}
        ids_b = {ann.id for ann in self.b.ann_mgr.items()}
        common = ids_a & ids_b
        
        # borrar series huérfanas
        for sid in list(self._series_by_id.keys()):
            if sid not in common:
                self._remove_one(sid)
        
        for sid in sorted(common):
            self._update_one(sid)
    
    def _remove_one(self, sid: str):
        if sid in self._series_by_id:
            del self._series_by_id[sid]
            self.view.clear_series(sid)
    
    def _update_one(self, sid: str):
        an_a = self.a.ann_mgr.get(sid)
        an_b = self.b.ann_mgr.get(sid)
        if an_a is None or an_b is None:
            self._remove_one(sid)
            return
        
        # asegurar pixeles
        if an_a.pixel_values is None or an_a.pixel_values.size == 0:
            an_a.capture_pixels(self.a)
        if an_b.pixel_values is None or an_b.pixel_values.size == 0:
            an_b.capture_pixels(self.b)
        
        pairs = _pair_pixels(an_a, an_b, self.max_points)
        if pairs is None or pairs.size == 0:
            self._remove_one(sid)
            return
        
        # actualizar serie
        pairs_list = pairs.tolist() if hasattr(pairs, 'tolist') else pairs
        self._series_by_id[sid] = pairs_list
        
        # actualizar vista
        color = QColor(an_a.color)
        color.setAlpha(128)  # 50% transparencia
        name = an_a.text if an_a.text else f"ann_{sid[:4]}"
        self.view.update_series(sid, pairs_list, color, name)
    
    def _save_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar imagen", "", "PNG (*.png)")
        if not path:
            return
        # Render directo de la vista
        pm = QPixmap(self.view.size())
        pm.fill(QColor(255, 255, 255))  # white
        painter = QPainter(pm)
        self.view.render(painter)
        painter.end()
        pm.save(path, "PNG")
