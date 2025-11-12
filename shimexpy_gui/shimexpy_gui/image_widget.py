"""
Custom widgets for the ShimExPy GUI.

This module contains the custom widgets used in the graphical interface.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QSizePolicy,
    QPushButton,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QTransform
)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QRectF, QPointF


class ImageDisplayView(QGraphicsView):
    """
    Widget based on QGraphicsView for displaying grayscale images (2D) and selecting ROI.
    Equivalent version to ImageDisplayLabel, but with native support for zoom,
    pan and vector graphics. Specifically optimized for 2D images.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Inicialización de escena ---
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.image_item = None
        
        # Inicialización del ROI (no añadido a la escena todavía)
        self.roi_rect_item = None

        # --- Estado de interacción ---
        self.drawing = False
        self.roi_enabled = False
        self.start_point = QPointF()
        self.end_point = QPointF()

        # --- Imagen original ---
        self.original_image = None
        self.window_min = None
        self.window_max = None
        
        # Control para prevenir actualizaciones anidadas
        self._updating = False

        # Estado de zoom
        self._zoom_factor = 1.0
        self._zoom_step = 1.25  # cada "tick" aumenta o reduce 25 %
        self._zoom_min = 0.2
        self._zoom_max = 20.0

        # Configuración básica de vista
        self.setRenderHints(
            self.renderHints() | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # ---------------------------------------------------------
    #  Mouse wheel event (zoom in / out)
    # ---------------------------------------------------------
    def wheelEvent(self, event):
        """Zoom with the mouse wheel (keeping centered)."""
        if self._updating:
            event.accept()  # Ignorar eventos durante actualizaciones
            return
            
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return
            
        # Punto bajo el cursor en coordenadas de escena (antes del zoom)
        # En PySide6, QWheelEvent usa position() en lugar de pos()
        mouse_point = self.mapToScene(event.position().toPoint())

        # Aumentar o reducir zoom
        if delta > 0:
            factor = self._zoom_step
        else:
            factor = 1 / self._zoom_step

        # Nuevo valor tentativo
        new_zoom = self._zoom_factor * factor
        if self._zoom_min <= new_zoom <= self._zoom_max:
            # Guardar la transformación actual
            old_pos = self.mapFromScene(mouse_point)
            
            # Aplicar zoom
            self.scale(factor, factor)
            self._zoom_factor = new_zoom
            
            # Recalcular la posición después del zoom
            new_pos = self.mapFromScene(mouse_point)
            delta = new_pos - old_pos
            
            # Ajustar la posición para mantener el punto bajo el cursor fijo
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            
        event.accept()

    # ---------------------------------------------------------
    #  Reset zoom when adjusting view (triggered by double-click)
    # ---------------------------------------------------------
    def reset_zoom(self):
        """Reset zoom to 1:1 (automatic fit to content)."""
        self.resetTransform()
        self._zoom_factor = 1.0
        if self.image_item is not None:
            self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)

    # -------------------------------------------------------------
    # Image scaling and display
    # -------------------------------------------------------------
    @staticmethod
    def rescale_to_8bit(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """
        Normalizes the 2D grayscale image to the 0-255 range.
        
        Args:
            img: Input 2D image (NumPy array)
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling
            
        Returns:
            Image normalized to uint8 (0-255)
        """
        if vmax <= vmin:
            return np.zeros_like(img, dtype=np.uint8)

        # Optimizado para imágenes 2D en escala de grises
        img_float = img.astype(np.float32, copy=False)
        norm = np.clip((img_float - vmin) / (vmax - vmin), 0, 1)
        return (norm * 255).astype(np.uint8)


    def _render_pixmap(self, image_8bit: np.ndarray):
        """
        Updates the image without losing zoom or moving the center when adjusting contrast.
        Optimized for 2D grayscale images.
        """
        if image_8bit is None or self._updating:
            return
            
        self._updating = True

        try:
            # Para imágenes 2D en escala de grises
            h, w = image_8bit.shape
            
            # Bytespp = 1 para escala de grises (un byte por píxel)
            bytes_per_line = w

            # Copia segura (desacoplar de NumPy)
            q_img = QImage(image_8bit.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8).copy()
            new_pixmap = QPixmap.fromImage(q_img)

            # Primera vez: crear pixmap y ajustar vista
            if self.image_item is None:
                self.image_item = QGraphicsPixmapItem(new_pixmap)
                self._scene.addItem(self.image_item)
                self._scene.setSceneRect(QRectF(0, 0, w, h))
                self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
                return

            # Guardar estado completo
            h_value = self.horizontalScrollBar().value()
            v_value = self.verticalScrollBar().value()
            transform_matrix = self.transform()
            
            # Guardar ROI si existe
            roi_rect = None
            roi_pos = None
            if self.roi_rect_item and self.roi_rect_item.scene() == self._scene:
                roi_rect = QRectF(self.roi_rect_item.rect())
                roi_pos = QPointF(self.roi_rect_item.pos())

            # Bloquear señales y actualizaciones temporalmente
            self.setUpdatesEnabled(False)
            
            # Actualizar pixmap
            self.image_item.setPixmap(new_pixmap)
            
            # Actualizar sceneRect si es necesario
            if (self._scene.width() != w or self._scene.height() != h):
                self._scene.setSceneRect(QRectF(0, 0, w, h))
                
            # Restaurar transformación EXACTAMENTE
            self.setTransform(transform_matrix)
            
            # Restaurar posición de la vista
            self.horizontalScrollBar().setValue(h_value)
            self.verticalScrollBar().setValue(v_value)
            
            # Restaurar ROI si existía
            if roi_rect is not None and roi_pos is not None and self.roi_rect_item is not None:
                self.roi_rect_item.setRect(roi_rect)
                self.roi_rect_item.setPos(roi_pos)
            
        except Exception as e:
            print(f"Error en _render_pixmap: {e}")
        finally:
            # Reactivar actualizaciones
            self.setUpdatesEnabled(True)
            self._updating = False



    def set_image(self, image: np.ndarray, preserve_view=True):
        """
        Loads and displays a 2D grayscale NumPy image.
        
        Args:
            image: 2D grayscale NumPy image
            preserve_view: If True, maintains the current zoom and position
        """
        if image is None:
            self._scene.clear()
            self.image_item = None
            return
            
        # Verify that it's a 2D image
        if image.ndim != 2:
            print("Error: Image must be 2D (grayscale)")
            return
        
        # Guardar transformación actual si es necesario
        current_transform = None
        if preserve_view and self.image_item is not None:
            current_transform = QTransform(self.transform())
        
        # Guardar posición central de la vista
        center_point = self.mapToScene(self.viewport().rect().center())
            
        self.original_image = image
        img_min = float(image.min())
        img_max = float(image.max())
        img_8bit = self.rescale_to_8bit(image, img_min, img_max)
        
        # Usar _render_pixmap que ya tiene lógica para preservar zoom
        self._render_pixmap(img_8bit)
        
        # Restaurar zoom y centrar si es necesario
        if preserve_view and current_transform is not None:
            self.setTransform(current_transform)
            self.centerOn(center_point)


    # -------------------------------------------------------------
    # Contrast/brightness adjustment from sliders
    # -------------------------------------------------------------
    @Slot(int, int)
    def set_window(self, slider_min: int, slider_max: int):
        """
        Linear window adjustment using sliders for 2D grayscale images.
        Maintains the current zoom and view position.
        
        Args:
            slider_min: Minimum slider value (0-255)
            slider_max: Maximum slider value (0-255)
        """
        if self.original_image is None:
            return
            
        # Calculamos rango de valores de la imagen 2D
        img_min = float(self.original_image.min())
        img_max = float(self.original_image.max())

        # Mapeo lineal de los valores del slider al rango real de la imagen
        vmin = img_min + (slider_min / 255.0) * (img_max - img_min)
        vmax = img_min + (slider_max / 255.0) * (img_max - img_min)
        
        # Evitar división por cero
        if vmax <= vmin:
            vmax = vmin + 1e-3 * (img_max - img_min)

        # Generar imagen de 8 bits para visualización
        img_8bit = self.rescale_to_8bit(self.original_image, vmin, vmax)
        
        # Delegar a _render_pixmap para mantener transformación y centro
        self._render_pixmap(img_8bit)

    # -------------------------------------------------------------
    # Interactive ROI
    # -------------------------------------------------------------
    def enable_roi(self, enable: bool):
        """Activates or deactivates ROI selection."""
        self.roi_enabled = enable
        if not enable:
            # Verificamos que el roi_rect_item existe y está en la escena antes de intentar eliminarlo
            if hasattr(self, 'roi_rect_item') and self.roi_rect_item is not None:
                # Verificar si el elemento pertenece a la escena antes de intentar eliminarlo
                if self.roi_rect_item.scene() == self._scene:
                    self._scene.removeItem(self.roi_rect_item)
            # Crear un nuevo elemento ROI
            self.roi_rect_item = QGraphicsRectItem()

    def mousePressEvent(self, event):
        if not self.roi_enabled or self.image_item is None:
            return super().mousePressEvent(event)

        self.drawing = True
        self.start_point = self.mapToScene(event.pos())
        self.end_point = self.start_point

        # Eliminar ROI anterior de forma segura
        if hasattr(self, 'roi_rect_item') and self.roi_rect_item is not None:
            if self.roi_rect_item.scene() == self._scene:
                self._scene.removeItem(self.roi_rect_item)
            
        # Crear un nuevo ROI
        self.roi_rect_item = QGraphicsRectItem(QRectF(self.start_point, self.end_point))
        self.roi_rect_item.setPen(QPen(QColor(255, 0, 0), 2))
        self.roi_rect_item.setBrush(QBrush(QColor(0, 0, 0, 60)))
        self._scene.addItem(self.roi_rect_item)

    def mouseMoveEvent(self, event):
        if not self.drawing or not self.roi_enabled:
            return super().mouseMoveEvent(event)

        self.end_point = self.mapToScene(event.pos())
        rect = QRectF(self.start_point, self.end_point).normalized()
        
        # Asegurarse de que roi_rect_item existe antes de usarlo
        if self.roi_rect_item is not None and hasattr(self.roi_rect_item, 'setRect'):
            self.roi_rect_item.setRect(rect)

    def mouseReleaseEvent(self, event):
        if not self.roi_enabled:
            return super().mouseReleaseEvent(event)

        self.drawing = False
        self.end_point = self.mapToScene(event.pos())
        
    def mouseDoubleClickEvent(self, event):
        """Reset zoom to fit content on double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_zoom()
        return super().mouseDoubleClickEvent(event)

    def get_roi_coords(self):
        """Returns the indices (slices) of the ROI in the original image coordinates."""
        if self.original_image is None or self.roi_rect_item is None:
            return None
            
        # Verificar que el ROI esté definido correctamente
        if not hasattr(self.roi_rect_item, 'rect'):
            return None
            
        rect = self.roi_rect_item.rect().normalized()
        
        # Verificar que el rectángulo tiene área
        if rect.width() <= 0 or rect.height() <= 0:
            return None

        x1, y1 = int(rect.left()), int(rect.top())
        x2, y2 = int(rect.right()), int(rect.bottom())

        # Asumimos imágenes 2D (escala de grises)
        h, w = self.original_image.shape
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        return (slice(y1, y2 + 1), slice(x1, x2 + 1))
        
    def get_image_or_roi(self):
        """
        Returns the complete image or the region of interest if enabled.
        
        Returns:
            tuple: (numpy_image, roi_slices)
            - numpy_image: np.ndarray with the complete image or ROI
            - roi_slices: tuple of slices if ROI was used, None if using complete image
        """
        if not self.roi_enabled or self.original_image is None:
            return self.original_image, None
            
        roi_coords = self.get_roi_coords()
        if roi_coords is None:
            return self.original_image, None
            
        try:
            # Extract the ROI subimage
            roi_image = self.original_image[roi_coords]
            return roi_image, roi_coords
        except:
            print("Error extracting ROI. Using complete image.")
            return self.original_image, None

    def center_view(self):
        """Centers the view on the image without changing the current zoom."""
        if self.image_item is not None:
            self.centerOn(self.image_item)
            
    def show_message(self, text: str):
        """Shows text in the scene, useful for error or status messages."""
        self._scene.clear()
        text_item = self._scene.addText(text)
        text_item.setDefaultTextColor(QColor("red"))
        self._scene.setSceneRect(text_item.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class Adjust_Sliders(QWidget):
    adjusts_changed = Signal(int, int)

    def __init__(self, min_value=0, max_value=255, parent=None):
        super().__init__(parent)

        # --- Sliders y etiquetas ---
        self.min_label = QLabel(str(min_value))
        self.max_label = QLabel(str(max_value))

        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 255)
        self.min_slider.setValue(min_value)

        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(0, 255)
        self.max_slider.setValue(max_value)

        # --- Layouts individuales (sin 'self' como padre) ---
        layout_control_min = QHBoxLayout()
        layout_control_min.addWidget(QLabel("Min:"))
        layout_control_min.addWidget(self.min_slider)
        layout_control_min.addWidget(self.min_label)

        layout_control_max = QHBoxLayout()
        layout_control_max.addWidget(QLabel("Max:"))
        layout_control_max.addWidget(self.max_slider)
        layout_control_max.addWidget(self.max_label)

        # --- Layout general del widget ---
        layout = QVBoxLayout(self)
        layout.addLayout(layout_control_min)
        layout.addLayout(layout_control_max)
        self.setLayout(layout)

        # --- Conexiones ---
        self.min_slider.valueChanged.connect(self.on_min_slider_changed)
        self.max_slider.valueChanged.connect(self.on_max_slider_changed)

        # --- Quitar bordes residuales y fondo opaco ---
        self.setStyleSheet("""
            QSlider, QLabel {
                border: none;
                background: transparent;
            }
        """)

    def on_min_slider_changed(self, value):
        self.adjusts_changed.emit(value, self.max_slider.value())
        self.min_label.setText(str(value))

    def on_max_slider_changed(self, value):
        self.adjusts_changed.emit(self.min_slider.value(), value)
        self.max_label.setText(str(value))


class ControlPanelWidget(QGroupBox):
    """
    Reusable control panel for ShimExPy.
    Contains all buttons, comboboxes, and processing parameters.
    """
    # Signals that will be emitted to MainWindow
    load_reference = Signal()
    load_sample = Signal()
    process_measurement = Signal()
    save_result = Signal()
    toggle_roi = Signal(bool)
    contrast_type_changed = Signal(str)
    grid_value_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__("Control Panel", parent)

        layout = QHBoxLayout(self)

        # --- Left column: image loading ---
        left_layout = QVBoxLayout()
        self.load_reference_btn = QPushButton("Load Reference")
        self.load_sample_btn = QPushButton("Load Sample")
        left_layout.addWidget(self.load_reference_btn)
        left_layout.addWidget(self.load_sample_btn)
        layout.addLayout(left_layout)

        # --- Center column: parameters ---
        middle_layout = QVBoxLayout()

        # Grid value
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid Value:"))
        self.grid_value_spinbox = QSpinBox()
        self.grid_value_spinbox.setRange(1, 100)
        self.grid_value_spinbox.setValue(5)
        grid_layout.addWidget(self.grid_value_spinbox)
        middle_layout.addLayout(grid_layout)

        # Contrast type
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast Type:"))
        self.contrast_type_combo = QComboBox()
        self.contrast_type_combo.addItems([
            "absorption",
            "horizontal_scattering",
            "vertical_scattering",
            "bidirectional_scattering",
            "horizontal_phasemap",
            "vertical_phasemap",
            "bidirectional_phasemap",
            "all"
        ])
        contrast_layout.addWidget(self.contrast_type_combo)
        middle_layout.addLayout(contrast_layout)

        layout.addLayout(middle_layout)

        # --- Right column: ROI + actions ---
        right_layout = QVBoxLayout()
        self.use_roi_check = QCheckBox("Use ROI")
        self.process_btn = QPushButton("Process Measurement")
        self.save_result_btn = QPushButton("Save Result")

        right_layout.addWidget(self.use_roi_check)
        right_layout.addWidget(self.process_btn)
        right_layout.addWidget(self.save_result_btn)
        layout.addLayout(right_layout)

        self.setLayout(layout)

        # --- Internal connections ---
        self.load_reference_btn.clicked.connect(self.load_reference.emit)
        self.load_sample_btn.clicked.connect(self.load_sample.emit)
        self.process_btn.clicked.connect(self.process_measurement.emit)
        self.save_result_btn.clicked.connect(self.save_result.emit)
        self.use_roi_check.toggled.connect(self.toggle_roi.emit)
        self.grid_value_spinbox.valueChanged.connect(self.grid_value_changed.emit)
        self.contrast_type_combo.currentTextChanged.connect(self.contrast_type_changed.emit)


class ImageDisplayWidget(QWidget):
    """
    Main container that includes:
      - ImageDisplayView (to display the image)
      - Adjust_Sliders (to adjust contrast/brightness)
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Main layout ---
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)  # 🔹 Small visual margin
        self.main_layout.setSpacing(12)  # 🔹 Clear space between image and sliders
        self.setLayout(self.main_layout)

        # --- Image ---
        self.image_display = ImageDisplayView()
        self.image_display.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.main_layout.addWidget(self.image_display, stretch=1)

        # --- Sliders (brightness/contrast adjustment) ---
        self.contrast_brightness_adjust = Adjust_Sliders(0, 255)
        self.contrast_brightness_adjust.setFixedHeight(70)
        self.contrast_brightness_adjust.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed
        )
        self.main_layout.addWidget(self.contrast_brightness_adjust, stretch=0)

        # --- Signals ---
        self.contrast_brightness_adjust.adjusts_changed.connect(
            self.image_display.set_window
        )

        # --- Suggested minimum sizes ---
        self.setMinimumHeight(600)
        self.setMinimumWidth(400)

    def sizeHint(self):
        """Size hint based on the current image (2D)."""
        if self.image_display.original_image is not None:
            h, w = self.image_display.original_image.shape  # Imagen 2D
            return QSize(w, h + 80)  # Añadimos espacio para los sliders
        return QSize(400, 480)


