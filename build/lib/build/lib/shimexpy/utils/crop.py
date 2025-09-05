"""
Utility functions for cropping images.
"""

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QMessageBox
import sys
from pathlib import Path


class crop_viewer(QWidget):
    """
    A simple GUI widget for selecting crop regions in an image.
    
    This class provides a basic interface for entering crop coordinates and
    displaying the result.
    
    Attributes
    ----------
    text_box : QLineEdit
        Text field for entering crop coordinates.
    button : QPushButton
        Button to confirm the crop selection.
    """
    def __init__(self):
        """Initialize the crop viewer widget."""
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        self.text_box = QLineEdit(self)
        self.text_box.setPlaceholderText("Escribe algo aquí...")
        layout.addWidget(self.text_box)

        self.button = QPushButton("Aceptar", self)
        self.button.clicked.connect(self.save_text)
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.setWindowTitle("Guarda Texto")
        self.resize(300, 100)

    def save_text(self):
        """Handle the button click event to save the entered text."""
        text = self.text_box.text()
        if text:
            print(f"Texto guardado: {text}")
            QMessageBox.information(self, "Guardado", "Texto guardado con éxito")
        else:
            QMessageBox.warning(self, "Advertencia", "El cuadro de texto está vacío")


def set_crop():
    """
    Initialize and display the crop viewer widget.
    
    This function is currently a placeholder for future implementation.
    It will be used to select crop regions for images.
    
    Returns
    -------
    tuple or None
        The selected crop region as (top, bottom, left, right) or None if canceled.
    """
    # Placeholder for future implementation
    pass
