# test_annotations.py
# Ejecuta:  python test_annotations.py
# Requisitos: tener en el mismo directorio annotation_item.py y annotation_manager.py

from __future__ import annotations
from typing import Tuple
import json
from pathlib import Path

from PySide6.QtGui import QColor
from annotation_manager import AnnotationManager


# ---------------------------
# Mock de figura tipo ResizableRectItem
# ---------------------------
class MockFigure:
    def __init__(self, shape: str = "rect",
                 rect: Tuple[float, float, float, float] = (0, 0, 10, 5),
                 color: QColor | None = None,
                 text: str = ""):
        self._shape = shape
        self._rect = rect
        self._color = color or QColor("black")
        self._text = text
        self._rotation = 0.0

    # Métodos de “apariencia”
    def set_color(self, color: QColor):
        self._color = color

    def set_label_text(self, text: str):
        self._text = text

    # Geometría
    def set_rect(self, x: float, y: float, w: float, h: float):
        self._rect = (x, y, w, h)

    def rect(self):
        # Objeto simple que imita QRectF (sólo lo necesario)
        class R:
            def __init__(self, x, y, w, h):
                self._x, self._y, self._w, self._h = x, y, w, h
            def x(self): return self._x
            def y(self): return self._y
            def width(self): return self._w
            def height(self): return self._h
        return R(*self._rect)

    # Rotación
    def set_rotation(self, rotation: float):
        self._rotation = float(rotation)

    def rotation(self) -> float:
        return self._rotation


# ---------------------------
# Factory para recrear figuras en importación
# ---------------------------
def figure_factory(shape: str, rect, color: QColor, text: str):
    return MockFigure(shape=shape, rect=rect, color=color, text=text)


def main():
    print("== Test de AnnotationManager/Item ==")
    mgr = AnnotationManager()

    # 1) Crear una figura y anotación
    fig1 = MockFigure(shape="rect", rect=(1, 2, 100, 50), color=QColor("red"), text="Label 1")
    ann1 = mgr.create(fig1, text="Test annotation", color=QColor("red"), shape="rect")
    print(f"Creado ann1: id={ann1.id} text={ann1.text} color={ann1.color.name()} rect={ann1.rect}")

    # 2) Actualizar color y texto
    mgr.update_color(ann1.id, QColor("blue"))
    mgr.update_text(ann1.id, "Updated label")
    print(f"Actualizado ann1: text={ann1.text} color={ann1.color.name()}")

    # 3) Añadimos otra anotación
    fig2 = MockFigure(shape="ellipse", rect=(10, 10, 40, 40), color=QColor("#80FF0000"), text="Circle-ish")
    ann2 = mgr.create(fig2, text="Second", color=QColor("#80FF0000"), shape="ellipse")
    ann2.figure.set_rotation(15)  # simulamos que la GUI rotó la figura
    mgr.refresh_from_figures()    # sincronizamos rotación/rect
    print(f"Creado ann2: id={ann2.id} shape={ann2.shape} rot={ann2.rotation} color={ann2.color.name(QColor.NameFormat.HexArgb)}")

    # 4) Exportar a JSON (string y archivo)
    json_str = mgr.to_json_str(indent=2)
    print("\n== JSON exportado ==")
    print(json_str)

    out_path = Path("annotations_export.json")
    mgr.to_json_file(str(out_path))
    print(f"\nGuardado en: {out_path.resolve()}")

    # 5) Importar desde string
    mgr_from_str = AnnotationManager.from_json_str(json_str, figure_factory=figure_factory)
    print(f"\nImportado desde string: {len(mgr_from_str)} anotaciones")
    for it in mgr_from_str.items():
        print(f"- {it.id} | {it.shape} | text='{it.text}' | color={it.color.name(QColor.NameFormat.HexArgb)} | rect={it.rect} | rot={it.rotation}")

    # 6) Importar desde archivo
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mgr_from_file = AnnotationManager.from_json_dict(data, figure_factory=figure_factory)
    print(f"\nImportado desde archivo: {len(mgr_from_file)} anotaciones")
    for it in mgr_from_file.items():
        print(f"- {it.id} | {it.shape} | text='{it.text}' | color={it.color.name(QColor.NameFormat.HexArgb)} | rect={it.rect} | rot={it.rotation}")

    # 7) Prueba de eliminación
    removed = mgr.remove_by_id(ann1.id)
    print(f"\nEliminado ann1 del manager original: {removed} | quedan {len(mgr)} anotaciones")

    print("\n== Test completado ==")


if __name__ == "__main__":
    main()
