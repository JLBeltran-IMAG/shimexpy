import json
from pathlib import Path
from typing import List
from ..logic.annotation_item import AnnotationItem


def save_annotations_to_json(annotations: List[AnnotationItem], filepath: str):
    data = [ann.to_dict() for ann in annotations]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_annotations_from_json(filepath: str) -> List[dict]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data  # Esto devuelve una lista de diccionarios para upsert_from_dict


    # # -------
    # # Updates
    # # -------
    # def refresh_from_figures(self) -> None:
    #     """Pull latest geometry/rotation from all live figures (e.g., after user interactions)."""
    #     for it in self._items.values():
    #         it.pull_from_figure()
    #     # One consolidated callback could be added if desired.


    # # --------------------------
    # # Import / Export
    # # --------------------------
    # def to_json_dict(self) -> Dict[str, Any]:
    #     self.refresh_from_figures()
    #     return {
    #         "version": 1,
    #         "annotations": [it.to_dict() for it in self._items.values()],
    #     }


    # def to_json_str(self, indent: Optional[int] = 2) -> str:
    #     return json.dumps(self.to_json_dict(), indent=indent)


    # def to_json_file(self, path: str, indent: Optional[int] = 2) -> None:
    #     with open(path, "w", encoding="utf-8") as f:
    #         f.write(self.to_json_str(indent=indent))


    # @staticmethod
    # def from_json_dict(data: Dict[str, Any], figure_factory: Optional[FigureFactory] = None) -> "AnnotationManager":
    #     mgr = AnnotationManager()
    #     anns = data.get("annotations", [])
    #     for d in anns:
    #         try:
    #             item = AnnotationItem.from_dict(d, figure_factory=figure_factory)
    #             mgr.add(item)
    #         except Exception:
    #             # Skip malformed entries but continue
    #             continue
    #     return mgr

    # @staticmethod
    # def from_json_str(s: str, figure_factory: Optional[FigureFactory] = None) -> "AnnotationManager":
    #     data = json.loads(s)
    #     return AnnotationManager.from_json_dict(data, figure_factory=figure_factory)

    # @staticmethod
    # def from_json_file(path: str, figure_factory: Optional[FigureFactory] = None) -> "AnnotationManager":
    #     with open(path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     return AnnotationManager.from_json_dict(data, figure_factory=figure_factory)

