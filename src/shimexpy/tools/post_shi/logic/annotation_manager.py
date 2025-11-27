import json
from typing import Any, Callable, Dict, Iterable, List, Optional
from PySide6.QtGui import QColor
from .annotation_item import AnnotationItem, FigureFactory


class AnnotationManager:
    """
    Container + basic operations for AnnotationItem objects.
    """

    def __init__(self) -> None:
        self._items: Dict[str, AnnotationItem] = {}

        # Optional callbacks (set from outside). Each receives the affected AnnotationItem.
        self.on_added: Optional[Callable[[AnnotationItem], None]] = None
        self.on_removed: Optional[Callable[[AnnotationItem], None]] = None
        self.on_cleared: Optional[Callable[[], None]] = None
        self.on_updated: Optional[Callable[[AnnotationItem], None]] = None

    # ----
    # CRUD
    # ----
    def add(self, item: AnnotationItem) -> None:
        self._items[item.id] = item
        if self.on_added:
            self.on_added(item)


    def create(self, figure: Any, text: str, color: QColor, shape: str = "rect") -> AnnotationItem:
        item = AnnotationItem(text=text, color=color, figure=figure, shape=shape)
        item.pull_from_figure()
        self.add(item)
        return item


    def remove_by_id(self, item_id: str) -> bool:
        item = self._items.pop(item_id, None)
        if item is None:
            return False
        if self.on_removed:
            self.on_removed(item)
        return True


    def remove(self, item: AnnotationItem) -> bool:
        return self.remove_by_id(item.id)


    def clear(self) -> None:
        self._items.clear()
        if self.on_cleared:
            self.on_cleared()


    def get(self, item_id: str) -> Optional[AnnotationItem]:
        return self._items.get(item_id)


    def items(self) -> Iterable[AnnotationItem]:
        return list(self._items.values())

    def ids(self) -> List[str]:
        """Return the list of annotation ids."""
        return list(self._items.keys())


    def __len__(self) -> int:
        return len(self._items)


    # ----------
    #  Mirroring
    # ----------
    def upsert_from_dict(self, data: Dict[str, Any], figure_factory: Optional[FigureFactory] = None) -> AnnotationItem:
        """
        Update-or-create by id and apply changes to the live figure.
        Assumes `data` comes from AnnotationItem.to_dict() in the paired container.
        """
        item_id = str(data["id"])
        existing = self._items.get(item_id)

        if existing is None:
            # Crear desde cero (y crear figura si hay factory)
            item = AnnotationItem.from_dict(data, figure_factory=figure_factory)
            self.add(item)
            return item

        # Actualizar campos lógicos
        existing.text = data["text"]
        existing.color = QColor(data["color"])
        existing.shape = data["shape"]
        existing.rect = tuple(data["rect"])
        existing.rotation = float(data["rotation"])

        if self.on_updated:
            self.on_updated(existing)

        return existing



