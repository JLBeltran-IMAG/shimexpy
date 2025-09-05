"""
Widgets personalizados para la GUI de ShimExPy.

Este módulo contiene los widgets personalizados utilizados en la interfaz gráfica.
"""

import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QPoint, QRect


class ImageDisplayLabel(QLabel):
    """
    Widget personalizado para mostrar imágenes con selección interactiva de ROI.
    
    Este widget extiende QLabel para permitir a los usuarios seleccionar una
    región de interés (ROI) haciendo clic y arrastrando sobre la imagen.
    """
    
    def __init__(self, parent=None):
        """Inicializar el widget."""
        super().__init__(parent)
        self.setMouseTracking(True)
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.drawing = False
        self.roi_enabled = False
        self.image_pixmap = None
        self.scaled_pixmap = None
        self.original_image = None
        self.scale_factor_w = 1.0
        self.scale_factor_h = 1.0
        
    def enable_roi(self, enable):
        """Habilitar o deshabilitar la selección de ROI."""
        self.roi_enabled = enable
        if not enable:
            self.start_point = QPoint()
            self.end_point = QPoint()
            self.update()
    
    def set_image(self, image):
        """
        Establecer la imagen a mostrar.
        
        Notas
        -----
        Existe una diferencia fundamental en cómo se representan las imágenes:
        - NumPy usa shape=(height, width) -> image.shape[0] = filas (y), image.shape[1] = columnas (x)
        - Qt usa (width, height) -> QImage(data, width, height, ...)
        
        Esta diferencia se debe tener en cuenta al convertir entre coordenadas del widget Qt
        y coordenadas de la matriz de NumPy.
        """
        self.original_image = image
        if image is None:
            self.image_pixmap = None
            self.scaled_pixmap = None
            self.setText("No image loaded")
            return
            
        # Normalize image to 0-255 for display
        normalized = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
        
        # Convert to QImage - QtImage usa (width, height) mientras NumPy usa (height, width)
        h, w = normalized.shape  # En NumPy: shape[0] = height (rows), shape[1] = width (columns)
        
        # En QImage: datos, ancho, alto, bytesPerLine, formato
        # bytesPerLine = width porque cada pixel es 1 byte (grayscale)
        q_img = QImage(normalized.data, w, h, w, QImage.Format.Format_Grayscale8)
        
        # Create pixmap
        self.image_pixmap = QPixmap.fromImage(q_img)
        self._update_scaled_pixmap()
        
        # Reset ROI points if any
        self.start_point = QPoint()
        self.end_point = QPoint()
        
        # Clear any text that might be displayed
        self.setText("")
        
        # Ensure the widget updates
        if hasattr(self, '_rect'):
            delattr(self, '_rect')
        self.update()
        
    def _update_scaled_pixmap(self):
        """
        Update the scaled pixmap based on the widget size.
        
        Este método escala la imagen original para ajustarla al tamaño del widget,
        manteniendo la relación de aspecto, y calcula los factores de escala necesarios
        para convertir coordenadas entre la imagen mostrada y la original.
        """
        if self.image_pixmap is None:
            return
            
        self.scaled_pixmap = self.image_pixmap.scaled(
            self.width(), 
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        
        # Calcular factores de escala
        if self.original_image is not None:
            # En NumPy, shape[0] = height (filas), shape[1] = width (columnas)
            img_height, img_width = self.original_image.shape
            
            # Calcular factores de escala: dimensión_pixmap / dimensión_original
            self.scale_factor_w = self.scaled_pixmap.width() / img_width if img_width > 0 else 1.0
            self.scale_factor_h = self.scaled_pixmap.height() / img_height if img_height > 0 else 1.0
        
        # Establecer el pixmap para mostrar la imagen
        self.setPixmap(self.scaled_pixmap)
        self.update()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self._update_scaled_pixmap()
    
    def mousePressEvent(self, ev):
        """Handle mouse press events."""
        if not self.roi_enabled or self.image_pixmap is None:
            return
            
        self.drawing = True
        self.start_point = ev.pos()
        self.end_point = ev.pos()
        self.update()
    
    def mouseMoveEvent(self, ev):
        """Handle mouse move events."""
        if not self.drawing or not self.roi_enabled:
            return
            
        self.end_point = ev.pos()
        self.update()
    
    def mouseReleaseEvent(self, ev):
        """Handle mouse release events."""
        if not self.roi_enabled or self.image_pixmap is None:
            return
            
        self.drawing = False
        self.end_point = ev.pos()
        self.update()
    
    def paintEvent(self, arg__1):
        """Handle paint events."""
        super().paintEvent(arg__1)
        
        if self.scaled_pixmap is None:
            return
        
        # Center the pixmap in the widget
        if not hasattr(self, '_rect'):
            # Calculate rect only once
            self._rect = QRect(
                (self.width() - self.scaled_pixmap.width()) // 2,
                (self.height() - self.scaled_pixmap.height()) // 2,
                self.scaled_pixmap.width(),
                self.scaled_pixmap.height()
            )
            
        if self.roi_enabled and not self.start_point.isNull() and not self.end_point.isNull():
            painter = QPainter(self)
            
            # Dibujamos un rectángulo semitransparente sobre el área no seleccionada
            # para resaltar mejor el ROI
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Calculamos las coordenadas del ROI
            x = min(self.start_point.x(), self.end_point.x())
            y = min(self.start_point.y(), self.end_point.y())
            width = abs(self.start_point.x() - self.end_point.x())
            height = abs(self.start_point.y() - self.end_point.y())
            
            # Crear un color semitransparente para el área no seleccionada
            overlay_color = QColor(0, 0, 0, 100)  # Negro con 39% de opacidad
            
            # Dibujar cuatro rectángulos para cubrir el área no seleccionada
            # 1. Parte superior
            painter.fillRect(0, 0, self.width(), y, overlay_color)
            # 2. Parte izquierda
            painter.fillRect(0, y, x, height, overlay_color)
            # 3. Parte derecha
            painter.fillRect(x + width, y, self.width() - (x + width), height, overlay_color)
            # 4. Parte inferior
            painter.fillRect(0, y + height, self.width(), self.height() - (y + height), overlay_color)
            
            # Dibujar el borde del ROI con un color más llamativo
            pen = QPen(QColor(255, 0, 0))  # Rojo
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(x, y, width, height)
            
            # Mostrar el tamaño del ROI
            if width > 60 and height > 20:  # Solo si hay espacio suficiente
                text = f"{width}x{height} px"
                painter.setPen(QColor(255, 255, 255))  # Texto blanco
                painter.drawText(x + 5, y + 20, text)
            
    def get_roi_coords(self):
        """
        Get the ROI coordinates in the original image space.
        
        Returns a tuple of slices: (row_slice, column_slice) that can be used to crop the image.
        In NumPy, this corresponds to image[row_slice, column_slice].
        
        IMPORTANTE: En NumPy las imágenes se almacenan como (rows, cols) -> (height, width),
        pero en Qt se trabaja como (width, height).
        
        Estas diferencias hacen que haya que tener cuidado al convertir entre coordenadas de pantalla
        y coordenadas de imagen para el ROI.
        """
        if self.original_image is None or not self.roi_enabled:
            return None
            
        # Calculate the offset if the image is centered
        if self.scaled_pixmap is None:
            return None
            
        # Calcular los offsets si la imagen está centrada en el widget
        offset_x = (self.width() - self.scaled_pixmap.width()) // 2
        offset_y = (self.height() - self.scaled_pixmap.height()) // 2
        
        # Obtener coordenadas del ROI en el espacio del widget (relativas a la imagen mostrada)
        start_widget_x = max(0, min(self.start_point.x() - offset_x, self.scaled_pixmap.width()))
        start_widget_y = max(0, min(self.start_point.y() - offset_y, self.scaled_pixmap.height()))
        end_widget_x = max(0, min(self.end_point.x() - offset_x, self.scaled_pixmap.width()))
        end_widget_y = max(0, min(self.end_point.y() - offset_y, self.scaled_pixmap.height()))
        
        # Obtener dimensiones de la imagen original en formato NumPy (height, width)
        img_height, img_width = self.original_image.shape
        
        # Convertir coordenadas del widget a coordenadas de la imagen original
        # Usamos los factores de escala que ya fueron calculados en _update_scaled_pixmap()
        start_img_x = int(start_widget_x / self.scale_factor_w)
        start_img_y = int(start_widget_y / self.scale_factor_h)
        end_img_x = int(end_widget_x / self.scale_factor_w)
        end_img_y = int(end_widget_y / self.scale_factor_h)
        
        # Asegurar que las coordenadas están dentro de los límites de la imagen
        start_img_x = max(0, min(img_width-1, start_img_x))
        start_img_y = max(0, min(img_height-1, start_img_y))
        end_img_x = max(0, min(img_width-1, end_img_x))
        end_img_y = max(0, min(img_height-1, end_img_y))
        
        # Crear slices en convención NumPy: imagen[fila_slice, columna_slice]
        # donde fila_slice corresponde a y (altura) y columna_slice a x (ancho)
        row_start = int(min(start_img_y, end_img_y))
        row_end = int(max(start_img_y, end_img_y))
        col_start = int(min(start_img_x, end_img_x))
        col_end = int(max(start_img_x, end_img_x))
        
        return (slice(row_start, row_end+1), slice(col_start, col_end+1))
