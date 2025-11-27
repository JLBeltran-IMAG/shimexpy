"""
Controller for managing processing functions in ShimExPy.

This module provides a controller that handles image processing
using ShimExPy functions for spatial harmonics analysis.
"""

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox

from shimexpy import get_harmonics, get_contrast, get_all_contrasts
from shimexpy import load_image


class ShimexController:
    """
    Controlador para las operaciones de procesamiento de ShimExPy.
    
    Esta clase maneja las operaciones de carga de imágenes, procesamiento,
    y gestión de resultados para la interfaz gráfica.
    """
    
    def __init__(self, main_window):
        """
        Inicializar el controlador.
        
        Parameters
        ----------
        main_window : MainWindow
            Ventana principal de la aplicación
        """
        self.main_window = main_window
        
        # Imágenes y resultados
        self.reference_image = None
        self.sample_image = None
        self.ref_block_grid = None
        self.ref_absorption = None
        self.ref_scattering = None
        self.ref_diff_phase = None
        self.contrast_result = None
    
    def load_reference(self):
        """
        Cargar la imagen de referencia.
        
        Returns
        -------
        bool
            True si se cargó correctamente, False en caso contrario
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Open Reference Image", "", "Images (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                self.reference_image = load_image(file_path)
                return True
            except Exception as e:
                self._show_error(f"Error loading reference image: {str(e)}")
                return False
        return False
    
    def load_sample(self):
        """
        Cargar la imagen de muestra.
        
        Returns
        -------
        bool
            True si se cargó correctamente, False en caso contrario
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Open Sample Image", "", "Images (*.tif *.tiff)"
        )
        
        if file_path:
            try:
                self.sample_image = load_image(file_path)
                return True
            except Exception as e:
                self._show_error(f"Error loading sample image: {str(e)}")
                return False
        return False
    
    def process_images(self, grid_value, contrast_type, roi_coords=None):
        """
        Procesar las imágenes de referencia y muestra.
        
        Parameters
        ----------
        grid_value : int
            Valor de la rejilla para el procesamiento
        contrast_type : str
            Tipo de contraste a calcular
        roi_coords : tuple, optional
            Coordenadas ROI si se está usando selección de región
            
        Returns
        -------
        dict
            Diccionario con los resultados del procesamiento
        """
        if self.reference_image is None or self.sample_image is None:
            self._show_error("Debe cargar una imagen de referencia y una imagen de muestra antes de procesar.")
            return None
        
        try:
            # Verificar dimensiones de las imágenes
            if self.sample_image.ndim != 2 or self.reference_image.ndim != 2:
                self._show_error(f"Las imágenes deben ser 2D. Muestra: {self.sample_image.ndim}D, Referencia: {self.reference_image.ndim}D")
                return None
                
            # Verificar tamaños mínimos
            if self.sample_image.shape[0] < 64 or self.sample_image.shape[1] < 64:
                self._show_error("La imagen de muestra es demasiado pequeña. Debe ser al menos 64x64 píxeles.")
                return None
                
            if self.reference_image.shape[0] < 64 or self.reference_image.shape[1] < 64:
                self._show_error("La imagen de referencia es demasiado pequeña. Debe ser al menos 64x64 píxeles.")
                return None
            
            # Procesar la imagen de referencia
            try:
                self.ref_absorption, self.ref_scattering, self.ref_diff_phase, self.ref_block_grid = get_harmonics(
                    self.reference_image, grid_value
                )
            except Exception as e:
                self._show_error(f"Error al procesar la imagen de referencia: {str(e)}")
                return None
            
            # Procesar la imagen de muestra según el tipo de contraste
            try:
                if contrast_type == "all":
                    abs_contrast, scat_contrast, diff_phase = get_all_contrasts(
                        self.sample_image, 
                        self.reference_image,
                        grid_value
                    )
                    
                    # Aplicar ROI si es necesario
                    if roi_coords:
                        try:
                            # Adaptar el ROI a las dimensiones de los resultados
                            abs_contrast = self._apply_roi_to_result(abs_contrast, roi_coords)
                            scat_contrast = self._apply_roi_to_result(scat_contrast, roi_coords)
                            diff_phase = self._apply_roi_to_result(diff_phase, roi_coords)
                        except Exception:
                            # Continuar con los resultados completos si hay error en el ROI
                            pass
                    
                    # Almacenar todos los resultados
                    self.contrast_result = {
                        "absorption": abs_contrast,
                        "scattering": scat_contrast,
                        "phasemap": diff_phase
                    }
                    
                    # Devolver el resultado de absorción por defecto
                    return {"result": abs_contrast, "title": "Absorption Contrast"}
                else:
                    # Seleccionar la referencia adecuada
                    if contrast_type == "absorption":
                        reference = self.ref_absorption
                    elif "scattering" in contrast_type:
                        reference = self.ref_scattering
                    elif "phasemap" in contrast_type:
                        reference = self.ref_diff_phase
                    else:
                        reference = self.ref_absorption
                    
                    # Obtener el contraste específico
                    contrast = get_contrast(
                        self.sample_image,
                        reference,
                        self.ref_block_grid,
                        contrast_type
                    )
                    
                    # Aplicar ROI si es necesario
                    if roi_coords:
                        try:
                            contrast = self._apply_roi_to_result(contrast, roi_coords)
                        except Exception:
                            # Continuar con el resultado completo si hay error en el ROI
                            pass
                    
                    # Almacenar el resultado
                    self.contrast_result = {contrast_type: contrast}
                    
                    return {"result": contrast, "title": f"{contrast_type.capitalize()} Contrast"}
            except Exception as e:
                self._show_error(f"Error al procesar: {str(e)}")
                return None
                
        except Exception as e:
            self._show_error(f"Error durante el procesamiento: {str(e)}")
            return None
            
    def get_result_for_contrast_type(self, contrast_type):
        """
        Obtener el resultado para un tipo de contraste específico.
        
        Parameters
        ----------
        contrast_type : str
            Tipo de contraste para el que se quiere obtener el resultado
            
        Returns
        -------
        dict
            Diccionario con el resultado y el título, o None si no hay resultado disponible
        """
        if not self.contrast_result:
            return None
            
        # Si tenemos el tipo de contraste solicitado
        if contrast_type in self.contrast_result:
            return {
                "result": self.contrast_result[contrast_type],
                "title": f"{contrast_type.capitalize()} Contrast"
            }
        elif contrast_type == "all" and "absorption" in self.contrast_result:
            return {
                "result": self.contrast_result["absorption"],
                "title": "Absorption Contrast"
            }
            
        # Si no tenemos el resultado solicitado, hay que reprocesar
        return None
    
    def _apply_roi_to_result(self, result, roi_coords):
        """
        Aplicar un ROI a un resultado de procesamiento.
        
        Parameters
        ----------
        result : ndarray
            Resultado del procesamiento
        roi_coords : tuple
            Coordenadas del ROI en la imagen original
            
        Returns
        -------
        ndarray
            Resultado recortado según el ROI
        """
        if roi_coords is None or result is None or self.sample_image is None:
            return result
            
        try:
            # Adaptar el ROI a las dimensiones del resultado
            h_orig, w_orig = self.sample_image.shape
            
            # Determinar la forma del resultado
            if result.ndim == 2:
                h_proc, w_proc = result.shape
            elif result.ndim == 3:
                h_proc, w_proc = result.shape[1:3]
            else:
                # Si no podemos determinar la forma, devolvemos el resultado sin modificar
                return result
            
            # Calcular factores de escala entre la imagen original y el resultado
            scale_h = h_proc / h_orig
            scale_w = w_proc / w_orig
            
            # Extraer los slices originales
            row_slice, col_slice = roi_coords
            
            # Calcular nuevos límites para el ROI escalado
            proc_row_start = int(row_slice.start * scale_h)
            proc_row_stop = int(row_slice.stop * scale_h)
            proc_col_start = int(col_slice.start * scale_w)
            proc_col_stop = int(col_slice.stop * scale_w)
            
            # Asegurar que los límites están dentro del rango válido
            proc_row_start = max(0, min(proc_row_start, h_proc-1))
            proc_row_stop = max(proc_row_start+1, min(proc_row_stop, h_proc))
            proc_col_start = max(0, min(proc_col_start, w_proc-1))
            proc_col_stop = max(proc_col_start+1, min(proc_col_stop, w_proc))
            
            # Crear los nuevos slices para el resultado
            proc_crop = (
                slice(proc_row_start, proc_row_stop),
                slice(proc_col_start, proc_col_stop)
            )
            
            # Aplicar el recorte según la dimensión del resultado
            if result.ndim == 2:
                return result[proc_crop]
            elif result.ndim == 3:
                return result[:, proc_crop[0], proc_crop[1]]
            else:
                return result
        except Exception:
            # En caso de error, devolver el resultado sin modificar
            return result
            
    def _show_error(self, message):
        """
        Mostrar un mensaje de error.
        
        Parameters
        ----------
        message : str
            Mensaje de error a mostrar
        """
        QMessageBox.critical(self.main_window, "Error", message)
