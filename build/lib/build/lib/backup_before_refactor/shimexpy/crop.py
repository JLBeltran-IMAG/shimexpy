import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QMessageBox
import sys
from pathlib import Path


class crop_viewer(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()

        self.text_box = QLineEdit(self)
        self.text_box.setPlaceholderText("Escribe algo aquí...")
        self.layout.addWidget(self.text_box)

        self.button = QPushButton("Aceptar", self)
        self.button.clicked.connect(self.save_text)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)
        self.setWindowTitle("Guarda Texto")
        self.resize(300, 100)

    def save_text(self):
        text = self.text_box.text()
        if text:
            print(f"Texto guardado: {text}")
            QMessageBox.information(self, "Guardado", "Texto guardado con éxito")
        else:
            QMessageBox.warning(self, "Advertencia", "El cuadro de texto está vacío")

if __name__ == "__main__":
    app = QApplication([])

    window = SimpleTextSaver()
    print(type(window))
    window.show()

    sys.exit(app.exec())
