"""
Entry point for the ShimExPy GUI.

This module provides the entry point to launch the graphical interface.
"""
import sys
import traceback
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMessageBox


try:
    from shimexpy_gui.mainwindow import MainWindow
except Exception as e:
    print(f"Error importing MainWindow: {str(e)}")
    traceback.print_exc()
    raise


def show_error_dialog(message, details):
    """Display an error dialog."""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Error in ShimExPy")
    msg_box.setText(message)
    msg_box.setDetailedText(details)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.exec()


def run_gui():
    """Start the GUI application."""
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # clean and neutral base style

        # --- Load QSS style ---
        qss_path = Path(__file__).parent / "assets" / "style.qss"
        if qss_path.exists():
            try:
                with qss_path.open("r", encoding="utf-8") as f:
                    app.setStyleSheet(f.read())
            except Exception as e:
                print(f"[WARN] Could not apply QSS style: {e}")
        else:
            print(f"[WARN] QSS style not found at: {qss_path}")

        # --- Create main window ---
        try:
            window = MainWindow()
            window.show()
            sys.exit(app.exec())
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"ERROR in the GUI: {str(e)}")
            if app:
                show_error_dialog(
                    f"An error has occurred in the application: {str(e)}",
                    error_details
                )

    except Exception as e:
        print(f"ERROR starting the application: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting ShimExPy GUI...")
    run_gui()
