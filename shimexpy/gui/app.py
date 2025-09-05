"""
Punto de entrada para la GUI de ShimExPy.

Este módulo proporciona el punto de entrada para iniciar la interfaz gráfica.
"""

import sys
import traceback
from PySide6.QtWidgets import QApplication, QMessageBox

# Importación de la ventana principal refactorizada
try:
    from shimexpy.gui.shimexpy_gui import MainWindow
except Exception as e:
    print(f"Error al importar MainWindow: {str(e)}")
    traceback.print_exc()
    raise


def show_error_dialog(message, details):
    """Mostrar un diálogo de error."""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Error en ShimExPy")
    msg_box.setText(message)
    msg_box.setDetailedText(details)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.exec()


def run_gui():
    """Iniciar la aplicación GUI."""
    try:
        app = QApplication(sys.argv)
        
        try:
            window = MainWindow()
            window.show()
            sys.exit(app.exec())
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"ERROR en la GUI: {str(e)}")
            
            # Mostrar un diálogo de error si la aplicación ya está en ejecución
            if app:
                show_error_dialog(
                    f"Se ha producido un error en la aplicación: {str(e)}",
                    error_details
                )
    except Exception as e:
        print(f"ERROR al iniciar la aplicación: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    run_gui()
