"""
Versión refactorizada de la ventana principal de ShimExPy.

Esta implementación mantiene exactamente la misma funcionalidad que main_window.py
pero con una estructura más modular y mejor organizada.
"""

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox, QMessageBox,
    QCheckBox, QGroupBox, QSplitter
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import sys
from pathlib import Path

from shimexpy.gui.image_widget import ImageDisplayLabel
from shimexpy.core.contrast import get_harmonics, get_contrast, get_all_contrasts
from shimexpy.io.file_io import load_image, save_image, save_block_grid, load_block_grid
from shimexpy.visualization.plot import plot_contrast


class MainWindow(QMainWindow):
    """
    Ventana principal para la aplicación ShimExPy GUI.
    
    Esta clase proporciona una interfaz gráfica para realizar análisis
    de imágenes de armonías espaciales en imágenes de rayos X.
    """
    
    def __init__(self):
        """Inicializar la ventana principal."""
        super().__init__()
        
        self.setWindowTitle("ShimExPy - Spatial Harmonics Imaging")
        self.resize(1200, 800)
        
        # Inicializar variables de instancia
        self.reference_image = None
        self.sample_image = None
        self.ref_block_grid = None
        self.ref_absorption = None
        self.ref_scattering = None
        self.ref_diff_phase = None
        self.contrast_result = None
        self.crop_region = None
        
        # Configurar la interfaz de usuario
        self._setup_ui()
    
    def _setup_ui(self):
        """Configurar los componentes de la interfaz de usuario."""
        # Widget y layout principal
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Panel de control en la parte superior
        control_panel = QGroupBox("Control Panel")
        control_layout = QHBoxLayout(control_panel)
        
        # Columna izquierda - Controles de carga de imágenes
        left_layout = QVBoxLayout()
        
        # Controles de imagen de referencia
        self.load_reference_btn = QPushButton("Load Reference")
        self.load_reference_btn.clicked.connect(self._load_reference)
        left_layout.addWidget(self.load_reference_btn)
        
        # Controles de imagen de muestra
        self.load_sample_btn = QPushButton("Load Sample")
        self.load_sample_btn.clicked.connect(self._load_sample)
        left_layout.addWidget(self.load_sample_btn)
        
        control_layout.addLayout(left_layout)
        
        # Columna central - Parámetros de procesamiento
        middle_layout = QVBoxLayout()
        
        # Controles de valor de rejilla
        grid_layout = QHBoxLayout()
        self.set_grid_value_label = QLabel("Grid Value:")
        grid_layout.addWidget(self.set_grid_value_label)
        
        self.grid_value_spinbox = QSpinBox()
        self.grid_value_spinbox.setRange(1, 100)
        self.grid_value_spinbox.setValue(5)  # Valor predeterminado
        grid_layout.addWidget(self.grid_value_spinbox)
        middle_layout.addLayout(grid_layout)
        
        # Controles de tipo de contraste
        contrast_layout = QHBoxLayout()
        self.contrast_type_label = QLabel("Contrast Type:")
        contrast_layout.addWidget(self.contrast_type_label)
        
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
        self.contrast_type_combo.currentIndexChanged.connect(self._update_result_display)
        contrast_layout.addWidget(self.contrast_type_combo)
        middle_layout.addLayout(contrast_layout)
        
        control_layout.addLayout(middle_layout)
        
        # Columna derecha - Controles de ROI y procesamiento
        right_layout = QVBoxLayout()
        
        # Casilla de verificación ROI
        self.use_roi_check = QCheckBox("Use Region of Interest (ROI)")
        self.use_roi_check.setToolTip("Enable to select a region of interest on the sample image")
        self.use_roi_check.toggled.connect(self._toggle_roi)
        right_layout.addWidget(self.use_roi_check)
        
        # Botón de proceso
        self.process_btn = QPushButton("Process Measurement")
        self.process_btn.setToolTip("Process both reference and sample images")
        self.process_btn.clicked.connect(self._process_measurement)
        self.process_btn.setEnabled(False)
        right_layout.addWidget(self.process_btn)
        
        control_layout.addLayout(right_layout)
        
        # Agregar panel de control al layout principal
        main_layout.addWidget(control_panel)
        
        # Área de visualización de imágenes con splitter para vista lado a lado
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Visualización de imagen de muestra (lado izquierdo)
        self.sample_display = ImageDisplayLabel()
        self.sample_display.setMinimumSize(400, 400)
        self.sample_display.setStyleSheet("border: 1px solid gray;")
        self.sample_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sample_display.setText("Sample Image\n(Load a sample image)")
        
        # Visualización de imagen de resultado (lado derecho)
        self.result_display = QLabel()
        self.result_display.setMinimumSize(400, 400)
        self.result_display.setStyleSheet("border: 1px solid gray;")
        self.result_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_display.setText("Contrast Result\n(Process to see result)")
        
        # Agregar visualizaciones al splitter
        self.splitter.addWidget(self.sample_display)
        self.splitter.addWidget(self.result_display)
        
        # Establecer tamaños iniciales
        self.splitter.setSizes([600, 600])
        
        # Agregar splitter al layout principal
        main_layout.addWidget(self.splitter)
        
        # Establecer el widget principal
        self.setCentralWidget(main_widget)
    
    def _toggle_roi(self, checked):
        """Habilitar o deshabilitar la selección de ROI."""
        self.sample_display.enable_roi(checked)
    
    def _load_reference(self):
        """Manejar la carga de una imagen de referencia."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Reference Image", "", "Images (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                self.reference_image = load_image(file_path)
                # Actualizar el estado del botón de proceso
                self._update_process_button_state()
            except Exception as e:
                self._show_error(f"Error loading reference image: {str(e)}")
    
    def _load_sample(self):
        """Manejar la carga de una imagen de muestra."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Sample Image", "", "Images (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                self.sample_image = load_image(file_path)
                # Mostrar la imagen de muestra
                self.sample_display.set_image(self.sample_image)
                # Actualizar el estado del botón de proceso
                self._update_process_button_state()
            except Exception as e:
                self._show_error(f"Error loading sample image: {str(e)}")
    
    def _update_process_button_state(self):
        """Actualizar el estado del botón de proceso según las imágenes cargadas."""
        self.process_btn.setEnabled(
            self.reference_image is not None and 
            self.sample_image is not None
        )
    
    def _adjust_image_size_for_fft(self, image):
        """
        Ajusta el tamaño de una imagen para que sea óptimo para FFT.
        
        FFT funciona mejor con tamaños de imagen que son potencias de 2 o 
        productos de pequeños números primos (2, 3, 5, 7).
        Este método redimensiona la imagen al siguiente tamaño óptimo cercano.
        
        Parameters
        ----------
        image : np.ndarray
            La imagen a redimensionar
            
        Returns
        -------
        np.ndarray
            Imagen redimensionada o la original si hay algún problema
        """
        try:
            from scipy.ndimage import zoom
            
            def next_fast_len(n):
                """Encuentra el siguiente número óptimo para FFT cercano a n."""
                # Potencias de 2
                next_power_of_2 = 2 ** np.ceil(np.log2(n))
                return int(next_power_of_2)
            
            h, w = image.shape
            new_h, new_w = h, w
            
            # Solo redimensionamos si el tamaño no es óptimo y la diferencia no es muy grande
            if h & (h-1) != 0:  # Verifica si h no es potencia de 2
                new_h = next_fast_len(h)
                # Si el nuevo tamaño es más de un 20% mayor, mejor usar el original
                if new_h > h * 1.2:
                    new_h = h
            
            if w & (w-1) != 0:  # Verifica si w no es potencia de 2
                new_w = next_fast_len(w)
                # Si el nuevo tamaño es más de un 20% mayor, mejor usar el original
                if new_w > w * 1.2:
                    new_w = w
            
            # Si el tamaño ya es óptimo o no cambia mucho, devolvemos la imagen original
            if new_h == h and new_w == w:
                return image
            
            # Aplicamos zoom para redimensionar
            scale_h = new_h / h
            scale_w = new_w / w
            
            resized = zoom(image, (scale_h, scale_w), order=1)
            return resized
        
        except Exception:
            return image
    
    def _process_measurement(self):
        """
        Procesar ambas imágenes: referencia y muestra.
        
        Este método procesa las imágenes de referencia y muestra usando el ROI si está activado,
        y actualiza la visualización con el contraste seleccionado.
        
        El manejo del ROI se hace completamente en este método, aplicándolo manualmente a las imágenes
        antes de enviarlas a las funciones de procesamiento. De esta forma, las funciones de procesamiento
        no necesitan preocuparse por el recorte.
        """
        if self.reference_image is None or self.sample_image is None:
            self._show_error("Debe cargar una imagen de referencia y una imagen de muestra antes de procesar.")
            return
        
        try:
            # Obtener la región de interés si está habilitada, pero SOLO para visualización
            # Para evitar problemas de compatibilidad con FFT, procesamos la imagen completa
            sample_crop = None
            
            if self.use_roi_check.isChecked():
                # get_roi_coords() devuelve (slice(row_start, row_end), slice(col_start, col_end))
                # Esto es compatible con la convención de NumPy: image[rows, cols]
                sample_crop = self.sample_display.get_roi_coords()
                if sample_crop:
                    roi_size = (sample_crop[0].stop - sample_crop[0].start, sample_crop[1].stop - sample_crop[1].start)
                    
                    # Verificamos que el ROI tenga un tamaño mínimo
                    if roi_size[0] < 32 or roi_size[1] < 32:
                        self._show_error("El área seleccionada es demasiado pequeña. Por favor, seleccione un área más grande (mínimo 32x32 píxeles).")
                        return
            
            # Verificamos que las imágenes tienen dimensiones razonables
            if self.sample_image.ndim != 2 or self.reference_image.ndim != 2:
                self._show_error(f"Las imágenes deben ser 2D. Muestra: {self.sample_image.ndim}D, Referencia: {self.reference_image.ndim}D")
                return
            
            # Verificamos que las imágenes no son demasiado pequeñas
            if self.sample_image.shape[0] < 64 or self.sample_image.shape[1] < 64:
                self._show_error("La imagen de muestra es demasiado pequeña. Debe ser al menos 64x64 píxeles.")
                return
                
            if self.reference_image.shape[0] < 64 or self.reference_image.shape[1] < 64:
                self._show_error("La imagen de referencia es demasiado pequeña. Debe ser al menos 64x64 píxeles.")
                return
            
            grid_value = self.grid_value_spinbox.value()
            
            # IMPORTANTE: Procesamos SIEMPRE la imagen completa para evitar problemas de compatibilidad
            # El ROI solo se usará para la visualización final
            try:
                self.ref_absorption, self.ref_scattering, self.ref_diff_phase, self.ref_block_grid = get_harmonics(
                    self.reference_image, grid_value
                )
            except Exception as e:
                self._show_error(f"Error al procesar la imagen de referencia: {str(e)}")
                return
            
            # Ahora procesamos la muestra (imagen completa)
            contrast_type = self.contrast_type_combo.currentText()
            
            try:
                if contrast_type == "all":
                    # Procesamos todos los tipos de contraste
                    abs_contrast, scat_contrast, diff_phase = get_all_contrasts(
                        self.sample_image,  # Imagen completa
                        self.reference_image,
                        grid_value
                    )
                    
                    # Siempre aplicamos ROI si está seleccionado
                    if sample_crop:
                        try:
                            # Mapeamos las coordenadas de ROI de la imagen original a la imagen procesada
                            # La imagen original es de tamaño (h_orig, w_orig) y la procesada es (h_proc, w_proc)
                            h_orig, w_orig = self.sample_image.shape
                            h_proc, w_proc = abs_contrast.shape
                            
                            # Calculamos factores de escala
                            scale_h = h_proc / h_orig
                            scale_w = w_proc / w_orig
                            
                            # Convertimos coordenadas del ROI de la imagen original a la procesada
                            proc_row_start = int(sample_crop[0].start * scale_h)
                            proc_row_stop = int(sample_crop[0].stop * scale_h)
                            proc_col_start = int(sample_crop[1].start * scale_w)
                            proc_col_stop = int(sample_crop[1].stop * scale_w)
                            
                            # Aseguramos que estén dentro de los límites
                            proc_row_start = max(0, min(proc_row_start, h_proc-1))
                            proc_row_stop = max(proc_row_start+1, min(proc_row_stop, h_proc))
                            proc_col_start = max(0, min(proc_col_start, w_proc-1))
                            proc_col_stop = max(proc_col_start+1, min(proc_col_stop, w_proc))
                            
                            # Creamos nuevo slice para la imagen procesada
                            proc_crop = (
                                slice(proc_row_start, proc_row_stop), 
                                slice(proc_col_start, proc_col_stop)
                            )
                            
                            # Aplicamos el ROI escalado
                            abs_contrast = abs_contrast[proc_crop]
                            
                            # Para scattering y phase, el recorte depende de la forma del array
                            if scat_contrast.ndim == 2:
                                scat_contrast = scat_contrast[proc_crop]
                            elif scat_contrast.ndim == 3:  # Si tiene dimensión extra (e.g., harmonics)
                                scat_contrast = scat_contrast[:, proc_crop[0], proc_crop[1]]
                                
                            if diff_phase.ndim == 2:
                                diff_phase = diff_phase[proc_crop]
                            elif diff_phase.ndim == 3:  # Si tiene dimensión extra
                                diff_phase = diff_phase[:, proc_crop[0], proc_crop[1]]
                        except Exception:
                            # Continuamos con los resultados completos si hay error en el ROI
                            pass
                    
                    self.contrast_result = {
                        "absorption": abs_contrast,
                        "scattering": scat_contrast,
                        "phasemap": diff_phase
                    }
                    
                    # Mostramos el contraste de absorción por defecto
                    self._display_array(abs_contrast, "Absorption Contrast")
                else:
                    # Determinamos qué referencia usar basado en el tipo de contraste
                    if contrast_type == "absorption":
                        reference = self.ref_absorption
                    elif "scattering" in contrast_type:
                        reference = self.ref_scattering
                    elif "phasemap" in contrast_type:
                        reference = self.ref_diff_phase
                    else:
                        # Por defecto, usamos absorción si el tipo no es reconocido
                        reference = self.ref_absorption
                    
                    # Obtenemos el contraste específico (imagen completa)
                    contrast = get_contrast(
                        self.sample_image,
                        reference,
                        self.ref_block_grid,
                        contrast_type
                    )
                    
                    # Si hay ROI, recortamos el resultado después del procesamiento
                    if sample_crop:
                        try:
                            # Mapeamos las coordenadas de ROI de la imagen original a la imagen procesada
                            h_orig, w_orig = self.sample_image.shape
                            h_proc, w_proc = contrast.shape if contrast.ndim == 2 else contrast.shape[1:3]
                            
                            # Calculamos factores de escala
                            scale_h = h_proc / h_orig
                            scale_w = w_proc / w_orig
                            
                            # Convertimos coordenadas del ROI de la imagen original a la procesada
                            proc_row_start = int(sample_crop[0].start * scale_h)
                            proc_row_stop = int(sample_crop[0].stop * scale_h)
                            proc_col_start = int(sample_crop[1].start * scale_w)
                            proc_col_stop = int(sample_crop[1].stop * scale_w)
                            
                            # Aseguramos que estén dentro de los límites
                            proc_row_start = max(0, min(proc_row_start, h_proc-1))
                            proc_row_stop = max(proc_row_start+1, min(proc_row_stop, h_proc))
                            proc_col_start = max(0, min(proc_col_start, w_proc-1))
                            proc_col_stop = max(proc_col_start+1, min(proc_col_stop, w_proc))
                            
                            # Creamos nuevo slice para la imagen procesada
                            proc_crop = (
                                slice(proc_row_start, proc_row_stop), 
                                slice(proc_col_start, proc_col_stop)
                            )
                            
                            # Manejar diferentes dimensiones posibles
                            if contrast.ndim == 2:
                                contrast = contrast[proc_crop]
                            elif contrast.ndim == 3:  # Si tiene dimensión extra (e.g., harmonics)
                                contrast = contrast[:, proc_crop[0], proc_crop[1]]
                                
                        except Exception:
                            # Continuamos con el resultado completo si hay error en el ROI
                            pass
                    
                    # Almacenamos el resultado
                    self.contrast_result = {contrast_type: contrast}
                    
                    # Mostramos el resultado
                    self._display_array(contrast, f"{contrast_type.capitalize()} Contrast")
            
            except Exception as e:
                error_msg = f"Error al procesar: {str(e)}"
                
                # Sugerencias específicas según el tipo de error
                if "setting an array element with a sequence" in str(e):
                    error_msg += "\n\nSugerencia: Hay un problema con el procesamiento de la imagen.\n"
                    error_msg += "Intente usar una imagen con diferente resolución o formato.\n"
                    error_msg += "Este error puede ocurrir cuando los arrays tienen formas incompatibles."
                elif "shape" in str(e).lower() and "broadcast" in str(e).lower():
                    error_msg += "\n\nSugerencia: Las formas de las imágenes no son compatibles.\n"
                    error_msg += "Asegúrese de que ambas imágenes tengan dimensiones similares."
                elif "memory" in str(e).lower():
                    error_msg += "\n\nSugerencia: No hay suficiente memoria para procesar estas imágenes.\n"
                    error_msg += "Intente con imágenes más pequeñas o cierre otras aplicaciones."
                
                self._show_error(error_msg)
                
        except Exception as e:
            self._show_error(f"Error durante el procesamiento: {str(e)}")
    
    def _update_result_display(self):
        """Actualizar la visualización del resultado según el tipo de contraste seleccionado."""
        if self.contrast_result is None:
            return
            
        contrast_type = self.contrast_type_combo.currentText()
        
        # Si tenemos el tipo de contraste seleccionado en nuestros resultados, lo mostramos
        if contrast_type in self.contrast_result:
            self._display_array(
                self.contrast_result[contrast_type],
                f"{contrast_type.capitalize()} Contrast"
            )
        elif contrast_type == "all" and "absorption" in self.contrast_result:
            # Si se selecciona "all" pero tenemos resultados individuales, mostramos absorción
            self._display_array(
                self.contrast_result["absorption"],
                "Absorption Contrast"
            )
        # Para otros casos donde no tenemos el contraste requerido, no hacemos nada
        # (requeriría reprocesar las imágenes)
    
    def _display_array(self, array, title=None):
        """Mostrar un array xarray o numpy en el visualizador de resultados."""
        if array is None:
            return
            
        try:
            # Convertir a array numpy si es un xarray
            if hasattr(array, 'values'):
                array = array.values
                
            # Manejar diferentes dimensiones de array
            if array.ndim == 2:
                # Array 2D - mostrar directamente
                self._display_image(array, title)
            elif array.ndim == 3:
                if array.shape[0] <= 8:  # Si es un array 3D con primera dimensión <= 8 (probablemente armonías)
                    # Para arrays 3D, mostramos la primera componente (generalmente la más importante)
                    self._display_image(array[0], f"{title} (Componente 1/{array.shape[0]})")
                else:
                    # Si la primera dimensión es grande, probablemente sea color o múltiples imágenes
                    self._display_image(array[0], f"{title} (Slice 1)")
            else:
                # Formato desconocido o dimensiones mayores
                # Intentamos mostrar la primera rebanada 2D que encontremos
                if array.ndim > 3:
                    # Para 4D o más, tomamos índices 0 hasta llegar a 2D
                    slice_indices = tuple([0] * (array.ndim - 2))
                    self._display_image(array[slice_indices], f"{title} (Rebanada multi-dimensional)")
                else:
                    # Como último recurso, mostramos la primera rebanada
                    self._display_image(array[0] if array.ndim > 2 else array, title)
        except Exception as e:
            self._show_error(f"Error al mostrar el resultado: {str(e)}")
    
    def _display_image(self, image, title=None):
        """Mostrar una imagen numpy en el visualizador de resultados."""
        try:
            # Verificar si la imagen es válida
            if image is None:
                self.result_display.setText("Error: Imagen no disponible")
                return
                
            # Verificar dimensiones
            if image.ndim != 2:
                # Si es 3D (posiblemente RGB), tomamos la primera capa
                if image.ndim == 3 and image.shape[2] in [1, 3, 4]:  # podría ser una imagen a color
                    image = image[:, :, 0]
                else:
                    while image.ndim > 2:
                        image = image[0]
            
            # Verificar valores anómalos
            has_nan = np.isnan(image).any()
            has_inf = np.isinf(image).any()
            
            if has_nan or has_inf:
                # Reemplazar NaN/Inf con 0
                image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Verificar si la imagen es constante (todos los valores iguales)
            if np.min(image) == np.max(image):
                # En este caso, normalizamos a un valor medio (128)
                normalized = np.full_like(image, 128, dtype=np.uint8)
            else:
                # Normalización simple sin alteraciones adicionales
                # Usamos directamente min y max de la imagen para preservar exactamente la escala original
                normalized = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
            
            # Convert to QImage - Manteniendo coherencia entre NumPy (h, w) y Qt (w, h)
            h, w = normalized.shape  # En NumPy: shape[0] = height (rows), shape[1] = width (columns)
            
            # En QImage, los parámetros son: data, width, height, bytesPerLine, format
            # bytesPerLine = width porque cada pixel es 1 byte (grayscale)
            q_img = QImage(normalized.data, w, h, w, QImage.Format.Format_Grayscale8)
            
            # Display in label
            pixmap = QPixmap.fromImage(q_img)
            self.result_display.setPixmap(pixmap.scaled(
                self.result_display.width(), 
                self.result_display.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            ))
            
            if title:
                self.result_display.setToolTip(title)
                
        except Exception as e:
            # Mostrar mensaje de error en el display
            self.result_display.setText(f"Error al mostrar imagen: {str(e)}")
            if title:
                self.result_display.setToolTip(f"Error: {title}")
    
    def _show_error(self, message):
        """Mostrar un mensaje de error."""
        QMessageBox.critical(self, "Error", message)


def run_gui():
    """Ejecutar la aplicación GUI de ShimExPy."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()
