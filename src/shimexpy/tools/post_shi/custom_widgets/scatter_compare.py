# custom_widgets/scatter_compare.py
from __future__ import annotations
import numpy as np
from typing import Optional, Callable, List, Tuple, Dict

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

from ..logic.annotation_item import AnnotationItem


class _MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class ScatterCompareWidget(QWidget):
    def __init__(self, container_a, container_b, max_points: int = 200_000, parent=None):
        super().__init__(parent)
        self.a = container_a
        self.b = container_b
        self.max_points = int(max_points)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # Canvas y toolbar
        self.canvas = _MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)

        # estado de límites
        self._xlim = None
        self._ylim = None
        self._limits_locked = False

        # fija límites iniciales a partir del rango de intensidades de ambas imágenes
        self._set_initial_limits()
        # escucha cambios de límites (cuando hagas zoom/pan con la toolbar)
        self.canvas.ax.callbacks.connect('xlim_changed', self._on_limits_changed)
        self.canvas.ax.callbacks.connect('ylim_changed', self._on_limits_changed)

        # Toolbar arriba, canvas debajo
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

        # Conectar a contenedores
        self._chain_callbacks()

        # Primer render
        self.refresh_plot()

    # -------------------- wiring --------------------
    def _chain_callbacks(self):
        # Conserva callbacks previos (p.ej. mirroring)
        prev_a_added = self.a.ann_mgr.on_added
        prev_a_updated = self.a.ann_mgr.on_updated
        prev_b_added = self.b.ann_mgr.on_added
        prev_b_updated = self.b.ann_mgr.on_updated

        def wrap(prev_cb: Optional[Callable[[AnnotationItem], None]], mine: Callable[[AnnotationItem], None]):
            def inner(ann: AnnotationItem):
                # Asegura que la anotación tenga pixel_values calculado en su contenedor
                self._ensure_pixels(ann)
                if prev_cb:
                    prev_cb(ann)
                mine(ann)
            return inner

        self.a.ann_mgr.on_added = wrap(prev_a_added, lambda ann: self.refresh_plot())
        self.a.ann_mgr.on_updated = wrap(prev_a_updated, lambda ann: self.refresh_plot())
        self.b.ann_mgr.on_added = wrap(prev_b_added, lambda ann: self.refresh_plot())
        self.b.ann_mgr.on_updated = wrap(prev_b_updated, lambda ann: self.refresh_plot())

        # Si tienes soporte de borrado, expón ann_mgr.on_removed y enchufa aquí similar:
        if hasattr(self.a.ann_mgr, "on_removed"):
            prev_a_removed = getattr(self.a.ann_mgr, "on_removed")
            self.a.ann_mgr.on_removed = wrap(prev_a_removed, lambda ann: self.refresh_plot())
        if hasattr(self.b.ann_mgr, "on_removed"):
            prev_b_removed = getattr(self.b.ann_mgr, "on_removed")
            self.b.ann_mgr.on_removed = wrap(prev_b_removed, lambda ann: self.refresh_plot())

    def _ensure_pixels(self, ann: AnnotationItem):
        """
        Garantiza que 'ann.pixel_values' esté calculado en el contenedor correcto.
        Detecta a cuál contenedor pertenece el 'ann' por presencia en su manager.
        """
        # ¿ann pertenece a A?
        if self.a.ann_mgr.get(ann.id) is ann:
            if ann.pixel_values is None or ann.pixel_values.size == 0:
                ann.capture_pixels(self.a)
        # ¿ann pertenece a B?
        if self.b.ann_mgr.get(ann.id) is ann:
            if ann.pixel_values is None or ann.pixel_values.size == 0:
                ann.capture_pixels(self.b)

    # -------------------- data & plot --------------------
    def _gather_series(self):
        """
        Recolecta series [(X,Y,color,label), ...] para todas las anotaciones
        con id presente en A y B. Usa los valores originales de las imágenes, sin normalización.
        """
        ids_a = {ann.id for ann in self.a.ann_mgr.items()}
        ids_b = {ann.id for ann in self.b.ann_mgr.items()}
        common = ids_a & ids_b
        if not common:
            return []

        series = []
        for aid in sorted(common):
            an_a = self.a.ann_mgr.get(aid)
            an_b = self.b.ann_mgr.get(aid)
            if not an_a or not an_b:
                continue

            if an_a.pixel_values is None or an_a.pixel_values.size == 0:
                an_a.capture_pixels(self.a)
            if an_b.pixel_values is None or an_b.pixel_values.size == 0:
                an_b.capture_pixels(self.b)

            va = an_a.pixel_values
            vb = an_b.pixel_values
            if va is None or vb is None or va.size == 0 or vb.size == 0:
                continue

            if va.ndim == 2:
                va = va.mean(axis=1)
            if vb.ndim == 2:
                vb = vb.mean(axis=1)

            n = min(va.size, vb.size)
            if n <= 0:
                continue

            X = va[:n].astype(float, copy=False)
            Y = vb[:n].astype(float, copy=False)

            if X.size > self.max_points:
                idx = np.random.RandomState(0).choice(X.size, size=self.max_points, replace=False)
                X, Y = X[idx], Y[idx]

            # color de la anotación
            qcol = an_a.color
            color = qcol.name()  # "#RRGGBB"
            label = an_a.text if an_a.text else f"ann_{aid[:4]}"

            series.append((X, Y, color, label))
        return series

    def refresh_plot(self):
        series = self._gather_series()
        ax = self.canvas.ax
        ax.clear()
        ax.set_autoscale_on(False)

        # Determinar los límites globales de X e Y para todas las series
        if series:
            all_x = np.concatenate([X for X, _, _, _ in series])
            all_y = np.concatenate([Y for _, Y, _, _ in series])
            min_x, max_x = np.min(all_x), np.max(all_x)
            min_y, max_y = np.min(all_y), np.max(all_y)
            # Hacer los límites iguales para ambos ejes para consistencia visual
            global_min = min(min_x, min_y)
            global_max = max(max_x, max_y)
            if global_min == global_max:
                # Evitar rango cero
                global_min -= 1
                global_max += 1
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)

        if not series:
            ax.set_title("Morphostructural analysis: There are no matching pairs")
            ax.set_xlabel("absorption")
            ax.set_ylabel("scattering")
            self.canvas.draw_idle()
            return

        for X, Y, color, label in series:
            ax.scatter(X, Y, s=5, alpha=0.5, c=color, label=label)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("absorption")
        ax.set_ylabel("scattering")
        ax.set_title("Morphostructural analysis")
        ax.legend(loc="best", fontsize="large")
        self.canvas.draw_idle()

    def _set_initial_limits(self):
        # Rango robusto (p1..p99) de ambas imágenes
        def prange(img):
            x = img.ravel().astype(float)
            p1, p99 = np.percentile(x, [1, 99])
            # por si la imagen es constante
            if p1 == p99:
                p1 -= 1.0
                p99 += 1.0
            return p1, p99

        p1a, p99a = prange(self.a.original_image)
        p1b, p99b = prange(self.b.original_image)
        lo = float(min(p1a, p1b))
        hi = float(max(p99a, p99b))

        # padding pequeño
        pad = 0.02 * (hi - lo)
        self._xlim = (lo - pad, hi + pad)
        self._ylim = (lo - pad, hi + pad)

        ax = self.canvas.ax
        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)
        ax.set_autoscale_on(False)  # <- clave: no autoscale
        self._limits_locked = True  # ya tenemos límites “fijos”

    def _on_limits_changed(self, _axis):
        """Guarda los límites cuando el usuario hace zoom/pan; se reaplicarán en cada refresh."""
        ax = self.canvas.ax
        self._xlim = ax.get_xlim()
        self._ylim = ax.get_ylim()
        self._limits_locked = True

