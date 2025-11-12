from PySide6.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QPushButton, QColorDialog, QMessageBox
from PySide6.QtGui import QColor




class LegendLabel(QDialog):
    """
    A dialog for entering a label and selecting a color.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.saved_label = None
        self.saved_color = None

        self.setWindowTitle("Add Label for Legend")
        self.setFixedSize(350, 50)

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        self.text_box = QLineEdit(self)
        self.text_box.setPlaceholderText("class ...")
        layout.addWidget(self.text_box)

        btn_add = QPushButton("Add", self)
        btn_add.clicked.connect(self.save_text)
        layout.addWidget(btn_add)

        btn_color = QPushButton("Color", self)
        btn_color.clicked.connect(self.select_color)
        layout.addWidget(btn_color)

    def save_text(self):
        text = self.text_box.text().strip()
        if text and self.saved_color is not None:
            self.saved_label = text
            QMessageBox.information(self, "Saved", "Label saved correctly")
            self.accept()

        elif text and self.saved_color is None:
            QMessageBox.warning(self, "Warning", "Color was not selected.")

        else:
            QMessageBox.warning(self, "Warning", "The text field is empty.")

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.saved_color = color


    def get_annotation(self) -> tuple[str, QColor] | None:
        """Returns the label and QColor if both were set correctly."""
        if self.saved_label and self.saved_color:
            return self.saved_label, self.saved_color
        return None



