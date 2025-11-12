"""
Refactored version of ShimExPy's main window.

This implementation maintains exactly the same functionality as main_window.py
but with a more modular and better organized structure.
"""

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout, 
    QFileDialog,
    QMessageBox,
    QSizePolicy
)
from PySide6.QtCore import Qt
import sys

from shimexpy_gui.image_widget import ImageDisplayWidget, ControlPanelWidget

from shimexpy import get_harmonics, get_contrast, get_all_contrasts
from shimexpy import load_image, save_image


class MainWindow(QMainWindow):
    """
    Main window for the ShimExPy GUI application.
    
    This class provides a graphical interface for performing analysis
    of spatial harmonics in X-ray images.
    """
    def __init__(self):
        """Initialize the main window."""
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
        """Configure the user interface components."""
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # --- Control panel (modular widget) ---
        self.control_panel = ControlPanelWidget()
        main_layout.addWidget(self.control_panel)

        # --- Display area ---
        self.sample_display = ImageDisplayWidget()
        self.result_display = ImageDisplayWidget()

        self.sample_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.result_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Horizontal layout for the two images
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.sample_display, 1)
        images_layout.addWidget(self.result_display, 1)

        # Add layout to the main one
        main_layout.addLayout(images_layout)

        main_layout.setStretch(0, 0)   # the control panel doesn't grow
        main_layout.setStretch(1, 1)   # the splitter occupies all the remaining space

        self.setCentralWidget(main_widget)

        # --- Signal connections ---
        self.control_panel.load_reference.connect(self._load_reference)
        self.control_panel.load_sample.connect(self._load_sample)
        self.control_panel.process_measurement.connect(self._process_measurement)
        self.control_panel.save_result.connect(self._save_result)
        self.control_panel.toggle_roi.connect(self._toggle_roi)
        self.control_panel.contrast_type_changed.connect(self._update_result_display)
        self.control_panel.grid_value_changed.connect(lambda v: None)


    def _toggle_roi(self, checked):
        """Enable or disable ROI selection."""
        self.sample_display.image_display.enable_roi(checked)

    def _load_reference(self):
        """Handle loading a reference image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Reference Image", "", "Images (*.tif *.tiff)"
        )

        if file_path:
            try:
                self.reference_image = load_image(file_path)
                # Update process button state
                self._update_process_button_state()
            except Exception as e:
                self._show_error(f"Error loading reference image: {str(e)}")
    
    def _load_sample(self):
        """Handle loading a sample image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Sample Image", "", "Images (*.tif *.tiff)"
        )

        if file_path:
            try:
                self.sample_image = load_image(file_path)
                # Display the sample image
                self.sample_display.image_display.set_image(self.sample_image)
                # Update process button state
                self._update_process_button_state()
            except Exception as e:
                self._show_error(f"Error loading sample image: {str(e)}")
    
    def _update_process_button_state(self):
        """Update the process button state based on loaded images."""
        self.control_panel.process_btn.setEnabled(
            self.reference_image is not None and 
            self.sample_image is not None
        )
    
    def _adjust_image_size_for_fft(self, image):
        """
        Adjusts the image size to be optimal for FFT.
        
        FFT works better with image sizes that are powers of 2 or
        products of small prime numbers (2, 3, 5, 7).
        This method resizes the image to the next closest optimal size.
        
        Parameters
        ----------
        image : np.ndarray
            The image to resize
            
        Returns
        -------
        np.ndarray
            Resized image or the original if there's a problem
        """
        try:
            from scipy.ndimage import zoom
            
            def next_fast_len(n):
                """Finds the next optimal number for FFT close to n."""
                # Potencias de 2
                next_power_of_2 = 2 ** np.ceil(np.log2(n))
                return int(next_power_of_2)
            
            h, w = image.shape
            new_h, new_w = h, w
            
            # Only resize if the size is not optimal and the difference is not too large
            if h & (h-1) != 0:  # Checks if h is not a power of 2
                new_h = next_fast_len(h)
                # If the new size is more than 20% larger, better use the original
                if new_h > h * 1.2:
                    new_h = h
            
            if w & (w-1) != 0:  # Checks if w is not a power of 2
                new_w = next_fast_len(w)
                # If the new size is more than 20% larger, better use the original
                if new_w > w * 1.2:
                    new_w = w
            
            # If the size is already optimal or doesn't change much, return the original image
            if new_h == h and new_w == w:
                return image
            
            # Apply zoom to resize
            scale_h = new_h / h
            scale_w = new_w / w
            
            resized = zoom(image, (scale_h, scale_w), order=1)
            return resized
        
        except Exception:
            return image
    
    def _process_measurement(self):
        """
        Process both images: reference and sample.
        
        This method processes the reference and sample images using the ROI if activated,
        and updates the display with the selected contrast.
        
        ROI handling is done completely in this method, applying it manually to the images
        before sending them to the processing functions. This way, the processing functions
        don't need to worry about cropping.
        """
        if self.reference_image is None or self.sample_image is None:
            self._show_error("You must load a reference image and a sample image before processing.")
            return
        
        try:
            # Get the region of interest if enabled, but ONLY for visualization
            # To avoid compatibility issues with FFT, we process the entire image
            sample_crop = None
            
            if self.control_panel.use_roi_check.isChecked():
                # get_roi_coords() returns (slice(row_start, row_end), slice(col_start, col_end))
                # This is compatible with NumPy convention: image[rows, cols]
                sample_crop = self.sample_display.image_display.get_roi_coords()
                if sample_crop:
                    roi_size = (sample_crop[0].stop - sample_crop[0].start, sample_crop[1].stop - sample_crop[1].start)
                    
                    # Verify that the ROI has a minimum size
                    if roi_size[0] < 32 or roi_size[1] < 32:
                        self._show_error("The selected area is too small. Please select a larger area (minimum 32x32 pixels).")
                        return
            
            # Verify that the images have reasonable dimensions
            if self.sample_image.ndim != 2 or self.reference_image.ndim != 2:
                self._show_error(f"Images must be 2D. Sample: {self.sample_image.ndim}D, Reference: {self.reference_image.ndim}D")
                return
            
            # Verify that the images are not too small
            if self.sample_image.shape[0] < 64 or self.sample_image.shape[1] < 64:
                self._show_error("The sample image is too small. It must be at least 64x64 pixels.")
                return
                
            if self.reference_image.shape[0] < 64 or self.reference_image.shape[1] < 64:
                self._show_error("The reference image is too small. It must be at least 64x64 pixels.")
                return

            grid_value = self.control_panel.grid_value_spinbox.value()

            # IMPORTANT: We ALWAYS process the complete image to avoid compatibility issues
            # The ROI will only be used for the final visualization
            try:
                self.ref_absorption, self.ref_scattering, self.ref_diff_phase, self.ref_block_grid = get_harmonics(
                    self.reference_image, grid_value
                )
            except Exception as e:
                self._show_error(f"Error processing the reference image: {str(e)}")
                return
            
            # Now we process the sample (complete image)
            contrast_type = self.control_panel.contrast_type_combo.currentText()

            try:
                if contrast_type == "all":
                    # Process all contrast types
                    abs_contrast, scat_contrast, diff_phase = get_all_contrasts(
                        self.sample_image,  # Imagen completa
                        self.reference_image,
                        grid_value
                    )
                    
                    # Always apply ROI if selected
                    if sample_crop:
                        try:
                            # Map ROI coordinates from the original image to the processed image
                            # The original image is size (h_orig, w_orig) and the processed is (h_proc, w_proc)
                            h_orig, w_orig = self.sample_image.shape
                            h_proc, w_proc = abs_contrast.shape
                            
                            # Calculate scale factors
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
                            
                            # Apply the scaled ROI
                            abs_contrast = abs_contrast[proc_crop]
                            
                            # For scattering and phase, the crop depends on the array shape
                            if scat_contrast.ndim == 2:
                                scat_contrast = scat_contrast[proc_crop]
                            elif scat_contrast.ndim == 3:  # If it has an extra dimension (e.g., harmonics)
                                scat_contrast = scat_contrast[:, proc_crop[0], proc_crop[1]]
                                
                            if diff_phase.ndim == 2:
                                diff_phase = diff_phase[proc_crop]
                            elif diff_phase.ndim == 3:  # If it has an extra dimension
                                diff_phase = diff_phase[:, proc_crop[0], proc_crop[1]]
                        except Exception:
                            # Continue with complete results if there's an error in the ROI
                            pass
                    
                    self.contrast_result = {
                        "absorption": abs_contrast,
                        "scattering": scat_contrast,
                        "phasemap": diff_phase
                    }
                    
                    # Show absorption contrast by default
                    self._display_array(abs_contrast, "Absorption Contrast")
                    # Enable save button
                    self.control_panel.save_result_btn.setEnabled(True)
                else:
                    # Determine which reference to use based on the contrast type
                    if contrast_type == "absorption":
                        reference = self.ref_absorption
                    elif "scattering" in contrast_type:
                        reference = self.ref_scattering
                    elif "phasemap" in contrast_type:
                        reference = self.ref_diff_phase
                    else:
                        # By default, use absorption if type is not recognized
                        reference = self.ref_absorption
                    
                    # Get the specific contrast (complete image)
                    contrast = get_contrast(
                        self.sample_image,
                        reference,
                        self.ref_block_grid,
                        contrast_type
                    )
                    
                    # If there's a ROI, crop the result after processing
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
                    # Habilitamos el botón de guardar
                    self.control_panel.save_result_btn.setEnabled(True)
            
            except Exception as e:
                error_msg = f"Error processing: {str(e)}"
                
                # Specific suggestions based on the type of error
                if "setting an array element with a sequence" in str(e):
                    error_msg += "\n\nSuggestion: There is a problem with the image processing.\n"
                    error_msg += "Try using an image with a different resolution or format.\n"
                    error_msg += "This error can occur when arrays have incompatible shapes."
                elif "shape" in str(e).lower() and "broadcast" in str(e).lower():
                    error_msg += "\n\nSuggestion: The shapes of the images are not compatible.\n"
                    error_msg += "Make sure both images have similar dimensions."
                elif "memory" in str(e).lower():
                    error_msg += "\n\nSuggestion: There is not enough memory to process these images.\n"
                    error_msg += "Try with smaller images or close other applications."
                
                self._show_error(error_msg)
                
        except Exception as e:
            self._show_error(f"Error during processing: {str(e)}")
    
    def _update_result_display(self):
        """Update the result display according to the selected contrast type."""
        if self.contrast_result is None:
            return

        contrast_type = self.control_panel.contrast_type_combo.currentText()

        # If we have the selected contrast type in our results, we show it
        if contrast_type in self.contrast_result:
            self._display_array(
                self.contrast_result[contrast_type],
                f"{contrast_type.capitalize()} Contrast"
            )
        elif contrast_type == "all" and "absorption" in self.contrast_result:
            # If "all" is selected but we have individual results, we show absorption
            self._display_array(
                self.contrast_result["absorption"],
                "Absorption Contrast"
            )
        # For other cases where we don't have the required contrast, we do nothing
        # (would require reprocessing the images)
    
    def _display_array(self, array, title=None):
        """Display an xarray or numpy array in the results viewer."""
        if array is None:
            return
            
        try:
            # Convert to numpy array if it's an xarray
            if hasattr(array, 'values'):
                array = array.values
                
            # Handle different array dimensions
            if array.ndim == 2:
                # 2D Array - display directly
                self._display_image(array, title)
            elif array.ndim == 3:
                if array.shape[0] <= 8:  # If it's a 3D array with first dimension <= 8 (probably harmonics)
                    # For 3D arrays, we show the first component (generally the most important)
                    self._display_image(array[0], f"{title} (Componente 1/{array.shape[0]})")
                else:
                    # If the first dimension is large, it's probably color or multiple images
                    self._display_image(array[0], f"{title} (Slice 1)")
            else:
                # Unknown format or higher dimensions
                # We try to show the first 2D slice we find
                if array.ndim > 3:
                    # For 4D or more, we take index 0 until reaching 2D
                    slice_indices = tuple([0] * (array.ndim - 2))
                    self._display_image(array[slice_indices], f"{title} (Rebanada multi-dimensional)")
                else:
                    # As a last resort, show the first slice
                    self._display_image(array[0] if array.ndim > 2 else array, title)
        except Exception as e:
            self._show_error(f"Error al mostrar el resultado: {str(e)}")


    def _display_image(self, image, title=None):
        """Display a numpy image in the results viewer (using ImageDisplayView)."""
        try:
            # Case: null image
            if image is None:
                self.result_display.image_display.show_message("Error: Imagen no disponible")
                return

            # Convert to numpy if it's xarray
            if hasattr(image, 'values'):
                image = image.values

            # Reduce to 2D if it comes with more dims
            if image.ndim != 2:
                if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                    image = image[:, :, 0]
                else:
                    while image.ndim > 2:
                        image = image[0]

            # Clean NaN/Inf
            if np.isnan(image).any() or np.isinf(image).any():
                image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

            # Display in the view
            self.result_display.image_display.set_image(image.astype(np.float32, copy=False))

            if title:
                self.result_display.image_display.setToolTip(title)

        except Exception as e:
            self.result_display.image_display.show_message(f"Error al mostrar imagen: {str(e)}")
            if title:
                self.result_display.image_display.setToolTip(f"Error: {title}")



    def _show_error(self, message):
        """Display an error message."""
        QMessageBox.critical(self, "Error", message)

    def _save_result(self):
        """
        Save the processed result as an image with the current contrast levels.
        Always saves the image with the min and max levels that are being displayed.
        """
        if self.contrast_result is None:
            self._show_error("There is no result to save. Process the images first.")
            return

        # Get the current contrast type
        contrast_type = self.control_panel.contrast_type_combo.currentText()

        # Get the corresponding array
        if contrast_type == "all":
            # If "all" is selected, we save the contrast that is currently being displayed
            if "absorption" in self.contrast_result:
                result_array = self.contrast_result["absorption"]
                current_type = "absorption"
            else:
                self._show_error("No valid result was found to save.")
                return
        else:
            # For any other type, we use the selected contrast
            if contrast_type in self.contrast_result:
                result_array = self.contrast_result[contrast_type]
                current_type = contrast_type
            else:
                self._show_error(f"The result of {contrast_type} was not found to save.")
                return

        # Open save file dialog
        default_name = f"shimexpy_{current_type}_result.tif"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            default_name,
            "TIFF Image (*.tif);;All Files (*.*)"
        )

        if file_path:
            try:
                # Get the original image from the results display
                original_image = self.result_display.image_display.original_image
                
                if original_image is not None:
                    # Make sure it's 2D to save it as an image
                    if original_image.ndim > 2:
                        original_image = original_image[0]
                    
                    # IMPORTANT! Here's the key: we apply the current slider levels
                    # to the original image before saving it
                    
                    # Get current slider values
                    slider_min = self.result_display.contrast_brightness_adjust.min_slider.value()
                    slider_max = self.result_display.contrast_brightness_adjust.max_slider.value()
                    
                    # Calculate the real range of the image
                    img_min = float(original_image.min())
                    img_max = float(original_image.max())
                    
                    # Calculate the real values that correspond to the slider positions
                    vmin = img_min + (slider_min / 255.0) * (img_max - img_min)
                    vmax = img_min + (slider_max / 255.0) * (img_max - img_min)
                    
                    # Avoid division by zero
                    if vmax <= vmin:
                        vmax = vmin + 1e-3 * (img_max - img_min)
                    
                    # Apply scaling to the original image using current contrast values
                    processed_image = np.clip((original_image - vmin) / (vmax - vmin), 0, 1)
                    
                    # Check if there is a shape difference with the original result
                    if processed_image.shape != result_array.shape and result_array.ndim == 2:
                        print(f"Note: The saved image has a different shape: {processed_image.shape} vs {result_array.shape}")
                    
                    # Save the processed image with the applied contrast levels
                    # but preserving the original floating point precision (not converting to 8 bits)
                    save_image(processed_image, file_path)
                    print(f"Image saved with levels: min={vmin:.3f}, max={vmax:.3f}")
                else:
                    # If for some reason there is no displayed image, use the original result
                    if result_array.ndim > 2:
                        result_array = result_array[0]
                    save_image(result_array, file_path)
                    print("Image saved with original levels (no image displayed)")
            except Exception as e:
                self._show_error(f"Error saving the result: {str(e)}")

