# custom_widgets/scatter_compare_qt.py
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QAction, QColor, QPixmap
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QToolBar, QFileDialog

from PySide6.QtCharts import (
    QChart, QChartView, QScatterSeries, QValueAxis
)

from ..logic.annotation_item import AnnotationItem


def _pair_pixels(an_a: AnnotationItem, an_b: AnnotationItem, max_points: int) -> Optional[np.ndarray]:
    """Devuelve Nx2 con pares (X=valA, Y=valB) promediando multicanal; None si no hay datos."""
    if an_a.pixel_values is None or an_b.pixel_values is None:
        return None
    va, vb = an_a.pixel_values, an_b.pixel_values
    if va.size == 0 or vb.size == 0:
        return None
    if va.ndim == 2:  # multicanal
        va = va.mean(axis=1)
    if vb.ndim == 2:
        vb = vb.mean(axis=1)

    n = min(va.size, vb.size)
    if n <= 0:
        return None
    va = va[:n].astype(float, copy=False)
    vb = vb[:n].astype(float, copy=False)
    if n > max_points:
        # muestreo aleatorio estable
        idx = np.random.RandomState(0).choice(n, size=max_points, replace=False)
        va = va[idx]
        vb = vb[idx]
    return np.stack([va, vb], axis=1)  # (N,2)


class ScatterCompareQt(QMainWindow):
    """
    Ventana de scatter con QtCharts, escuchando dos ImageContainer2D.
    - Un QScatterSeries por anotación (ID compartido en ambos contenedores).
    - Color = color de la anotación (alpha 0.5).
    - Leyenda = texto de la anotación.
    - Zoom/pan nativos y guardar PNG.
    """

    def __init__(self, container_a, container_b, max_points: int = 200_000, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Scatter Compare (QtCharts)")

        self.a = container_a
        self.b = container_b
        self.max_points = int(max_points)

        # ---- UI ----
        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        self.chart = QChart()
        self.chart.legend().setVisible(True)

        # Ejes (fijos; no autoscale)
        self.axis_x = QValueAxis()
        self.axis_y = QValueAxis()
        self.axis_x.setTitleText("Container A (intensidad)")
        self.axis_y.setTitleText("Container B (intensidad)")
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)

        # Rango inicial a partir de percentiles de ambas imágenes
        self._init_fixed_limits()

        self.view = QChartView(self.chart)
        self.view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)  # zoom con recuadro
        self.view.setRenderHint(self.view.renderHints())  # usar hints por defecto
        self.view.setMinimumSize(700, 500)                # tamaño F I J O inicial
        self.view.setMaximumHeight(2000)                  # deja al usuario redimensionar si quiere
        lay.addWidget(self.view)

        # Toolbar propia (guardar + reset)
        tb = QToolBar(self)
        act_save = QAction("Guardar PNG", self)
        act_reset = QAction("Reset View", self)
        act_save.triggered.connect(self._save_png)
        act_reset.triggered.connect(self._reset_view)
        tb.addAction(act_save)
        tb.addAction(act_reset)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        # series por id
        self._series_by_id: Dict[str, QScatterSeries] = {}

        # Escuchar eventos de anotaciones (sin pisar callbacks existentes)
        self._wire_to_managers()

        # Primer dibujado (por si ya habían anotaciones)
        self.refresh_all()

        # Performance: activar OpenGL si disponible (muy útil con muchas muestras)
        # Ojo: en algunos drivers/VMs podría no estar disponible.
        self._use_opengl = False
        try:
            # Verificar si OpenGL está disponible
            from PySide6.QtGui import QSurfaceFormat
            fmt = QSurfaceFormat()
            fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
            fmt.setVersion(2, 1)  # OpenGL 2.1 es suficiente para QCharts
            QSurfaceFormat.setDefaultFormat(fmt)
            
            # Intentar activar OpenGL en las series existentes
            for s in self._series_by_id.values():
                s.setUseOpenGL(True)
            self._use_opengl = True
        except Exception as e:
            print(f"OpenGL no disponible: {str(e)}. Usando renderizado software.")

    # -------------------- helpers de límites --------------------
    def _init_fixed_limits(self):
        def pr(img):
            x = img.astype(float).ravel()
            p1, p99 = np.percentile(x, [1, 99])
            if p1 == p99:
                p1 -= 1.0; p99 += 1.0
            return float(p1), float(p99)

        p1a, p99a = pr(self.a.original_image)
        p1b, p99b = pr(self.b.original_image)
        lo = min(p1a, p1b); hi = max(p99a, p99b)
        pad = 0.02 * (hi - lo)
        self._xlim = (lo - pad, hi + pad)
        self._ylim = (lo - pad, hi + pad)
        self.axis_x.setRange(*self._xlim)
        self.axis_y.setRange(*self._ylim)

    def _reset_view(self):
        self.axis_x.setRange(*self._xlim)
        self.axis_y.setRange(*self._ylim)

    def _save_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar imagen", "", "PNG (*.png)")
        if not path:
            return
        # Render directo del chartView
        pm = QPixmap(self.view.size())
        self.view.render(pm)
        pm.save(path, "PNG")

    # -------------------- wiring --------------------
    def _chain(self, prev: Optional[Callable[[AnnotationItem], None]], mine: Callable[[AnnotationItem], None]):
        def inner(ann: AnnotationItem):
            # asegurar píxeles computados por el contenedor correcto
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

    # -------------------- data & plot --------------------
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
        s = self._series_by_id.pop(sid, None)
        if s is not None:
            self.chart.removeSeries(s)

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

        # serie por id (crear si falta)
        s = self._series_by_id.get(sid)
        if s is None:
            s = QScatterSeries()
            s.setMarkerSize(4.0)
            s.setName(an_a.text if an_a.text else f"ann_{sid[:4]}")
            # color con alpha 0.5
            col = QColor(an_a.color)
            col.setAlphaF(0.5)
            s.setColor(col)            # relleno
            s.setBorderColor(Qt.GlobalColor.transparent)
            # enganchar a ejes y añadir al chart
            self.chart.addSeries(s)
            s.attachAxis(self.axis_x)
            s.attachAxis(self.axis_y)
            # OpenGL (si procede)
            try:
                s.setUseOpenGL(True)
            except Exception:
                pass
            self._series_by_id[sid] = s
        else:
            # actualizar nombre/color si cambiaron
            s.setName(an_a.text if an_a.text else f"ann_{sid[:4]}")
            col = QColor(an_a.color); col.setAlphaF(0.5)
            s.setColor(col); s.setBorderColor(Qt.GlobalColor.transparent)

        # cargar puntos
        # QScatterSeries no tiene replace(array) en PySide, así que limpiamos y agregamos
        s.clear()
        # convertir a QList<QPointF> implícitamente: append acepta QPointF
        for x, y in pairs:
            s.append(QPointF(float(x), float(y)))
