"""
Configuraciones para la GUI de ShimExPy.

Este módulo contiene definiciones de configuración para la GUI.
"""

# Colores y estilos
COLORS = {
    'primary': '#1e88e5',  # Azul principal
    'secondary': '#ff9800',  # Naranja secundario
    'success': '#4caf50',  # Verde para éxito
    'error': '#f44336',  # Rojo para errores
    'warning': '#ffeb3b',  # Amarillo para advertencias
    'background': '#f5f5f5',  # Fondo gris claro
    'text': '#212121',  # Texto oscuro
}

# Estilos de la aplicación
STYLESHEET = """
QMainWindow {
    background-color: #f5f5f5;
}

QPushButton {
    background-color: #1e88e5;
    color: white;
    border-radius: 4px;
    padding: 6px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #1976d2;
}

QPushButton:pressed {
    background-color: #0d47a1;
}

QPushButton:disabled {
    background-color: #bdbdbd;
    color: #757575;
}

QLabel {
    color: #212121;
}

QComboBox {
    border: 1px solid #bdbdbd;
    border-radius: 3px;
    padding: 3px;
}

QStatusBar {
    background-color: #e0e0e0;
    color: #212121;
}
"""

# Configuración de la ventana principal
WINDOW_CONFIG = {
    'title': 'ShimExPy',
    'width': 1200,
    'height': 800,
    'min_width': 800,
    'min_height': 600,
}

# Configuración de procesamiento
PROCESSING_CONFIG = {
    'default_contrast_type': 'amplitude',
    'default_filter': 'none',
}

# Mensajes de la aplicación
MESSAGES = {
    'welcome': 'Bienvenido a ShimExPy',
    'no_image': 'No hay imagen cargada',
    'processing_done': 'Procesamiento completado',
    'processing_error': 'Error durante el procesamiento',
    'save_success': 'Imagen guardada correctamente',
    'save_error': 'Error al guardar la imagen',
}

# Formatos de imagen soportados
IMAGE_FORMATS = [
    'tiff', 'tif', 'png', 'jpg', 'jpeg', 'bmp'
]
