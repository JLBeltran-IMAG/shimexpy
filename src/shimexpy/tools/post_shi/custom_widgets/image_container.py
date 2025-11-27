import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from functools import partial

from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QFileDialog,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QGraphicsScene,
    QGraphicsSimpleTextItem
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QMouseEvent,
    QColor,
    QBrush,
    QKeyEvent
)
from PySide6.QtCore import (
    QObject,
    Qt,
    QPointF,
    QEvent,
    QRectF,
    Signal
)


from ..logic.annotation_manager import AnnotationManager
from ..logic.annotation_item import AnnotationItem

from .resizable_rect_item import HandleItem, ResizableRectItem
from .zoomable_graphics import ZoomableGraphicsView
from .legend_label import LegendLabel
from .double_slider import DoubleSlider



class ImageContainer2D(QWidget):
    draw_mode_changed = Signal(str)

    def __init__(self, image: np.ndarray, type_of_contrast: str):
        super().__init__()

        # Image attributes
        self.original_image = image.astype(np.float32)
        self.image_min = self.original_image.min()
        self.image_max = self.original_image.max()
        self.type_of_contrast = type_of_contrast

        # About annotations
        self.annotation_mode = False
        self.annotation_temporarily_disabled = False

        self.annotation_items: list[ResizableRectItem] = []

        self.ann_mgr = AnnotationManager()
        self._ann_id_by_figure = {}

        self.current_annotation_text = None
        self.current_annotation_color = None

        # UI state
        self.status = False
        self.drawing = False

        # Geometry
        self.start_pos: QPointF = QPointF()
        self.rect_item: Optional[ResizableRectItem] | None = None
        self.moving_target: Optional[ResizableRectItem] | None = None

        self.moving_item = False
        self.last_mouse_pos = QPointF()

        self.figure_hovered = False

        self.rescaled_image = self.rescale_to_8bit(self.original_image, self.image_min, self.image_max)

        self.init_ui()


    def init_ui(self):
        layout = QVBoxLayout(self)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.viewport().installEventFilter(self)
        layout.addWidget(self.view)

        self.update_image_display()

        control_layout_all = QHBoxLayout()
        control_layout = QVBoxLayout()

        # Button to save the current view
        save_btn = QPushButton("Save view")
        save_btn.clicked.connect(self.save_view)
        control_layout.addWidget(save_btn)

        # Check to activate annotations
        self.annotation_group = QGroupBox("Draw Mode")
        self.annotation_group.setCheckable(True)
        self.annotation_group.setChecked(False)
        self.annotation_group.toggled.connect(self.toggle_annotation)

        # Radiobuttons
        self.rect_radio = QRadioButton("Rectangle")
        self.ellipse_radio = QRadioButton("Ellipse")

        # Connect buttons to annotation mode
        self.rect_radio.toggled.connect(self.update_draw_mode)
        self.ellipse_radio.toggled.connect(self.update_draw_mode)

        # Exclusive selection
        self.shape_button_group = QButtonGroup(self)
        self.shape_button_group.addButton(self.rect_radio)
        self.shape_button_group.addButton(self.ellipse_radio)
        self.shape_button_group.setExclusive(True)

        # Internal layout for the group box
        # ---------------------------------
        shape_layout = QVBoxLayout()
        shape_layout.addWidget(self.rect_radio)
        shape_layout.addWidget(self.ellipse_radio)
        self.annotation_group.setLayout(shape_layout)

        control_layout.addWidget(self.annotation_group)

        self.draw_mode: str | None = None


        # Double slider de 0 a 100 (relative percentage of the image range)
        # -----------------------------------------------------------------
        self.range_slider = DoubleSlider(minimum=0, maximum=100, initial_min=0, initial_max=100)
        self.range_slider.valueChanged.connect(self.update_display_from_sliders)
        control_layout_all.addWidget(self.range_slider)

        control_layout_all.addSpacing(30)

        control_layout_all.addLayout(control_layout)
        layout.addLayout(control_layout_all)


    def toggle_annotation(self, enabled: bool):
        if self.rect_item and self.rect_item.scene() is self.scene:
            self.scene.removeItem(self.rect_item)

        self.annotation_mode = enabled

        if not enabled:
            self.draw_mode = None
            self.draw_mode_changed.emit("none")  # <--- EMIT
        else:
            if self.rect_radio.isChecked():
                self.draw_mode = "rect"
            elif self.ellipse_radio.isChecked():
                self.draw_mode = "ellipse"

            self.draw_mode_changed.emit(self.draw_mode or "none")  # <--- EMIT


    def update_draw_mode(self):
        if self.rect_radio.isChecked():
            self.draw_mode = "rect"
        elif self.ellipse_radio.isChecked():
            self.draw_mode = "ellipse"
        else:
            self.draw_mode = None

        self.draw_mode_changed.emit(self.draw_mode or "none")  # <--- EMIT


    def save_view(self):
        home_dir = str(Path.home())
        filename, _ = QFileDialog.getSaveFileName(self, "Save View", home_dir, "PNG (*.png)")
        if filename:
            self.view.export_view(filename)
            path = Path(filename)
            if path.suffix == "":
                path = path.with_suffix(".png")
            self.view.export_view(str(path))


    def set_annotation_temporarily_disabled(self, disable: bool):
        self.annotation_temporarily_disabled = disable


    def set_hovered_status(self, inside: bool):
        self.figure_hovered = inside


    def open_label_dialog(self):
        """
        Opens a dialog for labeling the currently selected shape.

        This method handles the process of adding label and color to a drawn shape on the scene.
        It creates a text label associated with the shape and manages its positioning and appearance.

        The method performs the following operations:
        1. Opens a LegendLabel dialog for user input
        2. Sets the selected color to the shape
        3. Creates and positions a text label next to the shape
        4. Adds the shape to the list of annotations
        5. Creates an AnnotationItem and adds it to the annotation manager
        6. Notifies any registered observers of the new annotation

        Returns:
            None

        Notes:
            - Method will return early if no shape (rect_item) is currently selected
            - After successful labeling, the current rect_item is set to None
            - The shape's handles are hidden after labeling is complete
        """
        if not self.rect_item:
            return

        dialog = LegendLabel(self)
        if dialog.exec():
            result = dialog.get_annotation()
            if result:
                label, color = result

                # Set the pen color of the current shape
                self.rect_item.set_pen_color(color)

                # Add the text as QGraphicsSimpleTextItem
                text_item = QGraphicsSimpleTextItem(label)
                text_item.setBrush(QBrush(color))

                # Associate the text with the shape
                self.rect_item.set_label_item(text_item)

                # Position it near the shape (bottom left)
                scene_pos = self.rect_item.mapToScene(self.rect_item.rect.bottomLeft())
                text_item.setPos(scene_pos + QPointF(0, 5))
                self.scene.addItem(text_item)

                # Finalize this annotation
                self.rect_item.show_handles(False)
                self.annotation_items.append(self.rect_item)

                # At the end of open_label_dialog, after self.annotation_items.append(self.rect_item)
                ann = AnnotationItem(
                    figure=self.rect_item,
                    text=label,
                    color=color,
                    shape=self.draw_mode or "rect",
                )
                ann.pull_from_figure()
                ann.capture_pixels(self)
                stats = ann.pixel_stats()
                print("Pixel statistics:", stats)
                self.ann_mgr.add(ann)


                self._ann_id_by_figure[self.rect_item] = ann.id

                # Notify mirrors if using on_added
                if self.ann_mgr.on_added:
                    self.ann_mgr.on_added(ann)

                self._connect_item_signals(self.rect_item, ann.id)
                self.rect_item = None


    def keyPressEvent(self, event: QKeyEvent):
        if self.annotation_mode and event.key() == Qt.Key.Key_A:
            self.open_label_dialog()

        elif event.key() == Qt.Key.Key_Escape:
            if self.rect_item and self.rect_item.scene() is self.scene:
                self.scene.removeItem(self.rect_item)
                self.rect_item = None

        # elif event.key() == Qt.Key.Key_Delete:
        #     for it in list(self.scene.selectedItems()):
        #         if isinstance(it, ResizableRectItem):
        #             if it.label_item and it.label_item.scene():
        #                 self.scene.removeItem(it.label_item)
        #             self.scene.removeItem(it)
        #             # self._on_item_deleted(it)


    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched != self.view.viewport():
            return super().eventFilter(watched, event)

        if isinstance(event, QMouseEvent):
            scene_pos = self.view.mapToScene(event.position().toPoint())

            # Clic izquierdo → crear figura si está en modo anotación
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                # Siempre revisa si se hizo clic sobre una figura existente primero
                for fig in self.annotation_items:
                    if fig.contains_point(scene_pos):
                        fig.show_handles(True)
                        self.moving_item = True
                        self.moving_target = fig
                        self.last_mouse_pos = scene_pos
                        return True

                # Si no se hizo clic sobre figura, entra en modo creación si está habilitado
                if self.annotation_mode and not self.annotation_temporarily_disabled:
                    return self._handle_mouse_press(event)


            # Clic medio → mover figura
            elif event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.MiddleButton:
                return self._handle_mouse_press(event)

            elif event.type() == QEvent.Type.MouseMove:
                return self._handle_mouse_move(event)

            elif event.type() == QEvent.Type.MouseButtonRelease:
                return self._handle_mouse_release(event)

        elif event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            self.keyPressEvent(event)
            return True

        return super().eventFilter(watched, event)



    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = self.view.mapToScene(event.position().toPoint())
            self.drawing = True

            if self.rect_item and self.rect_item.scene() is self.scene:
                self.scene.removeItem(self.rect_item)
                self.rect_item = None

            self.rect_item = ResizableRectItem(QRectF(self.start_pos, self.start_pos), draw_ellipse=(self.draw_mode == "ellipse"))
            self.rect_item.hover_state_changed.connect(self.set_annotation_temporarily_disabled)
            self.rect_item.mouse_hover_state.connect(self.set_hovered_status)

            self.scene.addItem(self.rect_item)
            # self._connect_item_signals(self.rect_item)
            return True

        return False


    def _handle_mouse_move(self, event: QMouseEvent) -> bool:
        if self.drawing:
            # if not self.figure_hovered:
            #     return False  # Evita mover si el mouse ya no está sobre la figura

            current_pos = self.view.mapToScene(event.position().toPoint())
            rect = QRectF(self.start_pos, current_pos).normalized()

            if self.rect_item:
                self.rect_item.setRect(rect)

            return True

        elif self.moving_item:
            current_pos = self.view.mapToScene(event.position().toPoint())
            delta = current_pos - self.last_mouse_pos
            self.last_mouse_pos = current_pos

            if self.moving_target:
                self.moving_target.moveBy(delta.x(), delta.y())

            return True

        return False


    def _handle_mouse_release(self, event: QMouseEvent) -> bool:
        if self.drawing:
            self.drawing = False
            return True
        elif self.moving_item:
            self.moving_item = False
            self.moving_target = None
            return True
        return False


    @staticmethod
    def rescale_to_8bit(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Normalize of vmin-vmax to range 0-255."""
        if vmax - vmin == 0:
            return np.zeros_like(img, dtype=np.uint8)
        
        if img.ndim == 3:
            # For color images, normalize each channel independently
            normalized = np.zeros_like(img, dtype=np.uint8)
            for i in range(img.shape[2]):
                channel = img[..., i]
                channel_min = channel.min()
                channel_max = channel.max()
                if channel_max - channel_min > 0:
                    normalized[..., i] = (np.clip((channel - channel_min) / (channel_max - channel_min), 0, 1) * 255).astype(np.uint8)
            return normalized
        else:
            # For grayscale images
            return (np.clip((img - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)


    def update_image_display(self):
        img_8bit = self.rescaled_image
        h, w = img_8bit.shape[:2]
        
        if img_8bit.ndim == 2:
            # Grayscale image
            qimg = QImage(img_8bit.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # Color image (RGB)
            qimg = QImage(img_8bit.data, w, h, w * img_8bit.shape[2], QImage.Format.Format_RGB888)
            
        pixmap = QPixmap.fromImage(qimg)

        if not hasattr(self, "pixmap_item"):
            self.scene.clear()
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.view.setScene(self.scene)
            self.view.fitInView(self.pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.pixmap_item.setPixmap(pixmap)


    def update_display_from_sliders(self, values):
        slider_min, slider_max = values
        vmin = self.image_min + (slider_min / 100) * (self.image_max - self.image_min)
        vmax = self.image_min + (slider_max / 100) * (self.image_max - self.image_min)
        self.rescaled_image = self.rescale_to_8bit(self.original_image, vmin, vmax)
        self.update_image_display()





    # -------------------------- Signals ------------------------------------------------
    def set_draw_mode(self, mode: Optional[str]):
        """
        External method to set draw mode ('rect', 'ellipse', or None)
        and sync UI state accordingly.
        """
        if mode not in ("rect", "ellipse", None):
            return

        self.annotation_group.setChecked(mode is not None)

        if mode == "rect":
            self.rect_radio.setChecked(True)
        elif mode == "ellipse":
            self.ellipse_radio.setChecked(True)
        else:
            self.rect_radio.setAutoExclusive(False)
            self.ellipse_radio.setAutoExclusive(False)
            self.rect_radio.setChecked(False)
            self.ellipse_radio.setChecked(False)
            self.rect_radio.setAutoExclusive(True)
            self.ellipse_radio.setAutoExclusive(True)

        self.draw_mode = mode
        self.draw_mode_changed.emit(mode or "none")


    def _emit_update(self, ann_id: str, rect: QRectF):
        ann = self.ann_mgr.get(ann_id)
        if ann:
            ann.pull_from_figure()
            ann.capture_pixels(self)       # <-- NUEVO
            # (opcional) si usas puntos internos:
            # ann.refresh_point_values(self)
            if self.ann_mgr.on_updated:
                self.ann_mgr.on_updated(ann)


    def _connect_item_signals(self, item: ResizableRectItem, ann_id: str):
        item.geometry_committed.connect(partial(self._emit_update, ann_id))


    def mirror_annotation(self, source_ann: AnnotationItem):
        ann_dict = source_ann.to_dict()
        ann = self.ann_mgr.upsert_from_dict(ann_dict)

        if ann.figure is None:
            # --- Crear figura con rectángulo local y posición global ---
            x, y, w, h = ann.rect
            rect_local = QRectF(0, 0, w, h)

            figure = ResizableRectItem(rect_local, draw_ellipse=(ann.shape == "ellipse"))
            figure.setPos(x, y)  # Posiciona en escena
            figure.setRotation(ann.rotation)

            # Estilo y texto
            text_item = QGraphicsSimpleTextItem(ann.text)
            text_item.setBrush(ann.color)
            figure.set_pen_color(ann.color)
            figure.set_label_item(text_item)

            # Añadir a escena y enlazar
            ann.figure = figure
            self.annotation_items.append(figure)
            self.scene.addItem(figure)
            self.scene.addItem(text_item)
            self._ann_id_by_figure[figure] = ann.id

            self._connect_item_signals(figure, ann.id)

        else:
            # Ya existe la figura: actualizar geometría, posición y estilo
            x, y, w, h = ann.rect
            ann.figure.setRect(QRectF(0, 0, w, h))
            ann.figure.setPos(x, y)
            ann.figure.setRotation(ann.rotation)
            ann.figure.set_pen_color(ann.color)

            if ann.figure.label_item:
                ann.figure.label_item.setText(ann.text)
                ann.figure.label_item.setBrush(ann.color)
                ann.figure.update_label_position()


    def _apply_annotation_visuals(self, ann: AnnotationItem):
        fig = ann.figure
        if not fig:
            return
        fig.set_pen_color(ann.color)
        if fig.label_item:
            fig.label_item.setText(ann.text)
            fig.label_item.setBrush(ann.color)
        fig.setRect(QRectF(*ann.rect))
        fig.setRotation(ann.rotation)


    def update_annotation_text(self, item_id: str, new_text: str):
        ann = self.ann_mgr.get(item_id)
        if not ann or not ann.figure:
            return
        ann.text = new_text
        if ann.figure.label_item:
            ann.figure.label_item.setText(new_text)
        if self.ann_mgr.on_updated:
            self.ann_mgr.on_updated(ann)


    def update_annotation_color(self, item_id: str, new_color: QColor):
        ann = self.ann_mgr.get(item_id)
        if not ann or not ann.figure:
            return
        ann.color = new_color
        ann.figure.set_pen_color(new_color)
        if ann.figure.label_item:
            ann.figure.label_item.setBrush(new_color)
        if self.ann_mgr.on_updated:
            self.ann_mgr.on_updated(ann)


    # ----------- Pixels inside
    def scene_to_image_indices(self, pt_scene: QPointF) -> Tuple[int, int]:
        """
        Convierte un punto en escena a índices (row, col) del array de la imagen original.
        Usa el mapeo del QGraphicsPixmapItem para ser robusto a zoom/pan/fitInView.
        """
        if hasattr(self, "pixmap_item") and self.pixmap_item is not None:
            pt_item = self.pixmap_item.mapFromScene(pt_scene)  # coords de item ~ coords de píxel
            r = int(round(pt_item.y()))
            c = int(round(pt_item.x()))
        else:
            # Fallback: identidad
            r = int(round(pt_scene.y()))
            c = int(round(pt_scene.x()))
        return r, c


    def sample_pixel(self, r: int, c: int):
        """
        Devuelve el valor de (r,c) sobre self.original_image con verificación de límites.
        - Escala: tupla de int/float. Para float32 mantengo float (no fuerzo a int).
        - Grayscale -> (v,), RGB -> (r,g,b), etc.
        """
        img = self.original_image
        H, W = img.shape[:2]
        if r < 0 or r >= H or c < 0 or c >= W:
            return None

        val = img[r, c]
        if np.ndim(val) == 0:
            # escalar
            return (float(val),) if np.issubdtype(img.dtype, np.floating) else (int(val),)
        # vector
        arr = np.asarray(val)
        if np.issubdtype(arr.dtype, np.floating):
            return tuple(float(x) for x in arr.tolist())
        return tuple(int(x) for x in arr.tolist())







    # def _connect_item_signals(self, item: ResizableRectItem):
    #     def _find_ann():
    #         return self.ann_mgr.get(self._ann_id_by_figure.get(item))

    #     item.color_changed.connect(lambda c: self._on_item_color_changed(item, c))
    #     item.text_changed.connect(lambda t: self._on_item_text_changed(item, t))
    #     item.geometryCommitted.connect(lambda rect: self._on_item_geometry_changed(item, rect))
    #     item.deleted.connect(lambda: self._on_item_deleted(item))


    # def _on_item_color_changed(self, item, color):
    #     ann = self.ann_mgr.get(self._ann_id_by_figure.get(item))
    #     if ann:
    #         ann.color = color

    # def _on_item_text_changed(self, item, text):
    #     ann = self.ann_mgr.get(self._ann_id_by_figure.get(item))
    #     if ann:
    #         ann.text = text

    # def _on_item_geometry_changed(self, item, rect: QRectF):
    #     ann = self.ann_mgr.get(self._ann_id_by_figure.get(item))
    #     if ann:
    #         ann.rect = (rect.x(), rect.y(), rect.width(), rect.height())

    # def _on_item_deleted(self, item):
    #     ann_id = self._ann_id_by_figure.pop(item, None)
    #     if ann_id:
    #         self.ann_mgr.remove(ann_id)
