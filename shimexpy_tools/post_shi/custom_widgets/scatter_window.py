from PySide6.QtWidgets import QMainWindow
from .scatter_compare import ScatterCompareWidget


class ScatterWindow(QMainWindow):
    def __init__(self, container_a, container_b, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scatter Compare")
        self.scatter = ScatterCompareWidget(container_a, container_b)
        self.setCentralWidget(self.scatter)

        # Tamaño inicial fijo
        self.resize(800, 600)


