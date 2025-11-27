from dataclasses import dataclass, field
from ..custom_widgets.resizable_rect_item import ResizableRectItem
from .internal_point import InternalPoint
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4

import math
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtCore import QRectF, QPointF

from ..custom_widgets.resizable_rect_item import ResizableRectItem
from .internal_point import InternalPoint

if TYPE_CHECKING:
    # solo para hints; no ejecuta el import en runtime
    from custom_widgets.image_container import ImageContainer2D


# Factory para (re)crear la figura gráfica.
# Firma: (shape: str, rect: (x,y,w,h), color: QColor, text: str) -> ResizableRectItem
FigureFactory = Callable[[str, Tuple[float, float, float, float], QColor, str], ResizableRectItem]


def _hex_argb_to_qcolor(code: str) -> QColor:
    """Parse #AARRGGBB o #RRGGBB a QColor; negro si falla."""
    try:
        c = QColor(code)
        if not c.isValid():
            raise ValueError
        return c
    except Exception:
        return QColor(0, 0, 0, 255)


# Fallback de rasterización (skimage si está; si no, matplotlib.path)
try:
    from skimage.draw import polygon as sk_polygon
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

from matplotlib.path import Path as MplPath
from skimage.draw import polygon as sk_polygon

@dataclass
class AnnotationItem:
    """
    Representa una anotación lógica con figura (rect/ellipse), puntos internos,
    y cachés de píxeles contenidos por la figura.

    Atributos exportados:
      - id, text, color, shape, rect (x,y,w,h), rotation, points[ {id,name,u,v,visible,value} ]

    NOTA: 'figure' (ResizableRectItem) no se serializa.
    """
    text: str
    color: QColor
    figure: Optional[ResizableRectItem]
    shape: str = "rect"  # "rect" | "ellipse"
    rotation: float = 0.0
    rect: Tuple[float, float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
    id: str = field(default_factory=lambda: str(uuid4()))

    # --- puntos internos ---
    points: List[InternalPoint] = field(default_factory=list)

    # --- caches de muestreo de píxel ---
    pixel_values: Optional[np.ndarray] = None              # (N,) o (N,C)
    pixel_mask_bbox: Optional[np.ndarray] = None           # bool, shape: (hb, wb)
    pixel_bbox: Optional[Tuple[int, int, int, int]] = None # (r0, r1, c0, c1)
    pixel_bbox_data: Optional[np.ndarray] = None           # subimagen bbox (opcional)

    # -------------------------------------------------------------------------
    # Gestión de puntos
    # -------------------------------------------------------------------------
    def add_point(self, name: str, u: float, v: float) -> InternalPoint:
        """Agrega un punto interno en coordenadas normalizadas (u,v) dentro del rect."""
        pt = InternalPoint(id=str(uuid4()), name=name, u=float(u), v=float(v))
        self.points.append(pt)
        return pt

    def remove_point(self, point_id: str) -> bool:
        for i, p in enumerate(self.points):
            if p.id == point_id:
                self.points.pop(i)
                return True
        return False

    def clear_points(self) -> None:
        self.points.clear()

    # -------------------------------------------------------------------------
    # Sincronización desde la figura viva
    # -------------------------------------------------------------------------
    def pull_from_figure(self) -> None:
        """Actualiza rect y rotation desde la figura viva (si existe)."""
        if self.figure is None:
            return
        r: QRectF = self.figure.mapRectToScene(self.figure.rect)
        self.rect = (r.x(), r.y(), r.width(), r.height())
        self.rotation = float(self.figure.rotation())

    # -------------------------------------------------------------------------
    # Geometría y conversión (u,v) <-> escena
    # -------------------------------------------------------------------------
    def rect_tuple(self) -> Tuple[float, float, float, float]:
        """Fuente de verdad del rectángulo actual (x,y,w,h) en coords de escena."""
        if self.figure is not None:
            r: QRectF = self.figure.mapRectToScene(self.figure.rect)
            return (r.x(), r.y(), r.width(), r.height())
        return self.rect

    def current_rotation(self) -> float:
        """Rotación actual (figura si existe, si no, self.rotation)."""
        if self.figure is not None:
            return float(self.figure.rotation())
        return float(self.rotation)

    def _rotate_points(self, pts: np.ndarray, cx: float, cy: float, theta_deg: float) -> np.ndarray:
        """Rota puntos (M,2) alrededor de (cx,cy) por theta_deg (grados)."""
        if abs(theta_deg) < 1e-9:
            return pts
        th = math.radians(theta_deg)
        ct, st = math.cos(th), math.sin(th)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        rx = cx + dx * ct - dy * st
        ry = cy + dx * st + dy * ct
        return np.stack([rx, ry], axis=1)

    def _rect_polygon_scene(self) -> np.ndarray:
        """Polígono de 4 puntos del rectángulo rotado en escena (4,2)."""
        x0, y0, w, h = self.rect_tuple()
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        pts = np.array([
            [x0,     y0    ],
            [x0+w,   y0    ],
            [x0+w,   y0+h  ],
            [x0,     y0+h  ],
        ], dtype=float)
        return self._rotate_points(pts, cx, cy, self.current_rotation())

    def _ellipse_polygon_scene(self, n: int = 128) -> np.ndarray:
        """Aproxima la elipse rotada como polígono de n puntos (n,2) en escena."""
        x0, y0, w, h = self.rect_tuple()
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        t = np.linspace(0.0, 2.0 * math.pi, num=n, endpoint=False)
        ex = cx + 0.5 * w * np.cos(t)
        ey = cy + 0.5 * h * np.sin(t)
        pts = np.stack([ex, ey], axis=1)
        return self._rotate_points(pts, cx, cy, self.current_rotation())

    def _shape_polygon_scene(self) -> np.ndarray:
        """Polígono de la figura actual en escena."""
        if self.shape == "ellipse":
            return self._ellipse_polygon_scene()
        return self._rect_polygon_scene()

    def _uv_to_scene(self, u: float, v: float) -> QPointF:
        """Convierte (u,v) normalizados del rect a punto en escena (con rotación)."""
        x0, y0, w, h = self.rect_tuple()
        px = x0 + u * w
        py = y0 + v * h
        theta = math.radians(self.current_rotation())
        if abs(theta) < 1e-9:
            return QPointF(px, py)
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        dx, dy = px - cx, py - cy
        rx = cx + dx * math.cos(theta) - dy * math.sin(theta)
        ry = cy + dx * math.sin(theta) + dy * math.cos(theta)
        return QPointF(rx, ry)

    def _scene_to_uv(self, pt: QPointF) -> Tuple[float, float]:
        """Convierte punto de escena a (u,v) normalizados del rect (des-rotando)."""
        x0, y0, w, h = self.rect_tuple()
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        theta = math.radians(self.current_rotation())
        dx, dy = pt.x() - cx, pt.y() - cy
        ux = dx * math.cos(-theta) - dy * math.sin(-theta) + cx
        uy = dx * math.sin(-theta) + dy * math.cos(-theta) + cy
        u = (ux - x0) / (w if w != 0 else 1.0)
        v = (uy - y0) / (h if h != 0 else 1.0)
        return u, v

    def points_scene_positions(self) -> List[Tuple[InternalPoint, QPointF]]:
        """[(InternalPoint, QPointF_en_escena)] solo de puntos visibles."""
        return [(p, self._uv_to_scene(p.u, p.v)) for p in self.points if p.visible]

    # -------------------------------------------------------------------------
    # Captura de píxeles de la figura (máscara + valores)
    # -------------------------------------------------------------------------
    def _polygon_image_indices(self, container: "ImageContainer2D") -> Tuple[np.ndarray, np.ndarray]:
        """
        Convierte polígono en escena a coordenadas de imagen (col, row) float.
        Si tu mapeo escena->imagen no es 1:1, reemplaza el bucle usando tu helper.
        """
        poly_scene = self._shape_polygon_scene()  # (M,2) en escena
        cols = []
        rows = []
        for x, y in poly_scene:
            # Reemplaza con: c, r = container.scene_to_image_point(QPointF(x,y))
            c = float(x)
            r = float(y)
            cols.append(c)
            rows.append(r)
        return np.array(cols, dtype=float), np.array(rows, dtype=float)

    def capture_pixels(self, container: "ImageContainer2D") -> None:
        """
        Rasteriza la figura a máscara y extrae valores de píxel del contenedor.
        Guarda:
          - pixel_values: np.ndarray (N,) o (N,C)
          - pixel_mask_bbox: bool (hb,wb)
          - pixel_bbox: (r0,r1,c0,c1)
          - pixel_bbox_data: subimagen bbox (opcional)
        """
        image = container.original_image  # np.ndarray (H,W[,C])
        H, W = image.shape[:2]

        poly_c, poly_r = self._polygon_image_indices(container)

        c_min = max(int(np.floor(poly_c.min())), 0)
        c_max = min(int(np.ceil (poly_c.max())) + 1, W)
        r_min = max(int(np.floor(poly_r.min())), 0)
        r_max = min(int(np.ceil (poly_r.max())) + 1, H)
        if c_min >= c_max or r_min >= r_max:
            self.pixel_values = np.empty((0,), dtype=image.dtype)
            self.pixel_mask_bbox = None
            self.pixel_bbox = None
            self.pixel_bbox_data = None
            return

        poly_c_local = poly_c - c_min
        poly_r_local = poly_r - r_min
        hb, wb = (r_max - r_min), (c_max - c_min)

        yy, xx = np.mgrid[0:hb, 0:wb]
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        path = MplPath(np.stack([poly_c_local, poly_r_local], axis=1))
        mask = path.contains_points(pts).reshape(hb, wb)

        subimg = image[r_min:r_max, c_min:c_max]

        if subimg.ndim == 2:
            vals = subimg[mask]        # (N,)
        else:
            vals = subimg[mask, :]     # (N, C)

        self.pixel_values = vals
        self.pixel_mask_bbox = mask
        self.pixel_bbox = (r_min, r_max, c_min, c_max)
        self.pixel_bbox_data = subimg  # quita si no quieres RAM extra

    # -------------------------------------------------------------------------
    # Remuestreo de valores para puntos internos (si los usas)
    # -------------------------------------------------------------------------
    def refresh_point_values(self, container: "ImageContainer2D") -> None:
        """
        Re-muestrea el valor de píxel de cada punto interno.
        Requiere en el contenedor:
          - scene_to_image_indices(QPointF) -> (row, col)
          - sample_pixel(row, col) -> Tuple[int,...] | None
        """
        for p, scene_pt in self.points_scene_positions():
            r, c = container.scene_to_image_indices(scene_pt)
            p.value = container.sample_pixel(r, c)

    def pixel_stats(self) -> Optional[dict]:
        """
        Devuelve stats básicas de self.pixel_values.
        Grayscale: float; Multicanal: listas por canal.
        """
        vals = self.pixel_values
        if vals is None or vals.size == 0:
            return None

        if vals.ndim == 1:
            return {
                "n": int(vals.size),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p50": float(np.percentile(vals, 50)),
            }
        else:
            # (N, C)
            return {
                "n": int(vals.shape[0]),
                "min": np.min(vals, axis=0).astype(float).tolist(),
                "max": np.max(vals, axis=0).astype(float).tolist(),
                "mean": np.mean(vals, axis=0).astype(float).tolist(),
                "std": np.std(vals, axis=0).astype(float).tolist(),
                "p50": np.percentile(vals, 50, axis=0).astype(float).tolist(),
            }

    # -------------------------------------------------------------------------
    # Serialización
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa datos lógicos. Llama antes a pull_from_figure() si quieres
        geometría/rotación al día.
        """
        return {
            "version": 1,
            "id": self.id,
            "text": self.text,
            "color": self.color.name(QColor.NameFormat.HexArgb),
            "shape": self.shape,
            "rect": list(self.rect),
            "rotation": float(self.rotation),
            "points": [
                {
                    "id": p.id,
                    "name": p.name,
                    "u": float(p.u),
                    "v": float(p.v),
                    "visible": bool(p.visible),
                    "value": list(p.value) if p.value is not None else None,
                }
                for p in self.points
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        figure_factory: Optional[FigureFactory] = None,
    ) -> "AnnotationItem":
        """
        Reconstruye un AnnotationItem desde un diccionario exportado. Si se aporta
        figure_factory, se crea y adjunta la figura viva.
        """
        item_id = str(data.get("id") or str(uuid4()))
        text = data.get("text", "")
        color = _hex_argb_to_qcolor(data.get("color", "#FF000000"))
        shape = data.get("shape", "rect")
        rect_list = data.get("rect", [0.0, 0.0, 0.0, 0.0])
        rect_t = (float(rect_list[0]), float(rect_list[1]), float(rect_list[2]), float(rect_list[3]))
        rotation = float(data.get("rotation", 0.0))

        figure = None
        if figure_factory is not None:
            figure = figure_factory(shape, rect_t, color, text)

        ann = cls(
            text=text,
            color=color,
            figure=figure,
            shape=shape,
            rect=rect_t,
            rotation=rotation,
            id=item_id,
        )

        # Restaurar puntos (si existen)
        for p in data.get("points", []):
            ip = InternalPoint(
                id=str(p.get("id") or str(uuid4())),
                name=str(p.get("name", "")),
                u=float(p.get("u", 0.5)),
                v=float(p.get("v", 0.5)),
                value=tuple(p["value"]) if p.get("value") is not None else None,
                visible=bool(p.get("visible", True)),
            )
            ann.points.append(ip)

        return ann




