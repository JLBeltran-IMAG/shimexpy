from PySide6.QtWidgets import QWidget, QSlider, QLabel, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, Signal



class DoubleSlider(QWidget):
    valueChanged = Signal(tuple)

    def __init__(self, minimum=0, maximum=255, initial_min=0, initial_max=255, parent=None):
        super().__init__(parent)

        self.min_val = initial_min
        self.max_val = initial_max
        self.minimum = minimum
        self.maximum = maximum

        # Sliders
        self.slider_min = QSlider(Qt.Orientation.Horizontal)
        self.slider_max = QSlider(Qt.Orientation.Horizontal)

        for slider in (self.slider_min, self.slider_max):
            slider.setMinimum(minimum)
            slider.setMaximum(maximum)

        self.slider_min.setValue(initial_min)
        self.slider_max.setValue(initial_max)

        # Labels of sliders
        self.label_min = QLabel(str(initial_min))
        self.label_max = QLabel(str(initial_max))

        # Layouts
        sliders_layout = QVBoxLayout()
        sliders_layout.addWidget(self.slider_min)
        sliders_layout.addWidget(self.slider_max)

        labels_layout = QHBoxLayout()
        labels_layout.addWidget(self.label_min)
        labels_layout.addStretch()
        labels_layout.addWidget(self.label_max)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(sliders_layout)
        main_layout.addLayout(labels_layout)

        # Connections
        self.slider_min.valueChanged.connect(self.update_values)
        self.slider_max.valueChanged.connect(self.update_values)

        self.setLayout(main_layout)

    def update_values(self):
        new_min = self.slider_min.value()
        new_max = self.slider_max.value()

        # Avoid invalid range
        if new_min > new_max:
            if self.sender() == self.slider_min:
                self.slider_min.setValue(self.slider_max.value())
                return
            else:
                self.slider_max.setValue(self.slider_min.value())
                return

        self.min_val = new_min
        self.max_val = new_max

        self.label_min.setText(str(self.min_val))
        self.label_max.setText(str(self.max_val))

        self.valueChanged.emit((self.min_val, self.max_val))

    def setRange(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        self.slider_min.setMinimum(minimum)
        self.slider_min.setMaximum(maximum)
        self.slider_max.setMinimum(minimum)
        self.slider_max.setMaximum(maximum)

    def getValues(self):
        return (self.min_val, self.max_val)


