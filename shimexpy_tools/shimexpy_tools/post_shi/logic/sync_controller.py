"""Synchronization controller for image containers."""
from typing import Any, Callable
from PySide6.QtCore import QObject, Signal

from ..custom_widgets.image_container import ImageContainer2D
from .annotation_item import AnnotationItem

class SyncController(QObject):
    """Controls synchronization between two image containers."""
    
    sync_state_changed = Signal(bool)  # Emitted when sync state changes
    
    def __init__(self, container_a: ImageContainer2D, container_b: ImageContainer2D):
        super().__init__()
        self.container_a = container_a
        self.container_b = container_b
        self._suppress_mirror = False
        self._suppress_sync = False
        self.setup_connections()
    
    def setup_connections(self):
        """Set up all synchronization connections."""
        self._setup_annotation_mirroring()
        self._setup_draw_mode_sync()
    
    def _setup_annotation_mirroring(self):
        """Set up bi-directional annotation mirroring."""
        self.container_a.ann_mgr.on_added = lambda ann: self._mirror_annotation(self.container_a, self.container_b, ann)
        self.container_a.ann_mgr.on_updated = lambda ann: self._mirror_annotation(self.container_a, self.container_b, ann)
        self.container_b.ann_mgr.on_added = lambda ann: self._mirror_annotation(self.container_b, self.container_a, ann)
        self.container_b.ann_mgr.on_updated = lambda ann: self._mirror_annotation(self.container_b, self.container_a, ann)
    
    def _mirror_annotation(self, src: ImageContainer2D, dst: ImageContainer2D, ann: AnnotationItem):
        """Mirror an annotation from source to destination container."""
        if self._suppress_mirror:
            return
        try:
            self._suppress_mirror = True
            dst.mirror_annotation(ann)
        finally:
            self._suppress_mirror = False
    
    def _setup_draw_mode_sync(self):
        """Set up draw mode synchronization between containers."""
        self.container_a.annotation_group.toggled.connect(
            lambda enabled: self._sync_draw_mode(self.container_a, self.container_b, enabled)
        )
        self.container_b.annotation_group.toggled.connect(
            lambda enabled: self._sync_draw_mode(self.container_b, self.container_a, enabled)
        )
    
    def _sync_draw_mode(self, source: ImageContainer2D, target: ImageContainer2D, enabled: bool):
        """Sync draw mode state between containers."""
        if self._suppress_sync:
            return
        if enabled and target.annotation_group.isChecked():
            try:
                self._suppress_sync = True
                target.annotation_group.setChecked(False)
            finally:
                self._suppress_sync = False
