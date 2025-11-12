"""Main module for morphostructural analysis."""
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from argparse import ArgumentParser, Namespace
import logging

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, 
    QVBoxLayout, QMenuBar, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from .custom_widgets.image_container import ImageContainer2D
from .custom_widgets.scatter_window import ScatterWindow
from .custom_widgets.scatter_compare_simple import ScatterCompareSimple
from .logic.sync_controller import SyncController
from .data.annotation_data import save_annotations_to_json, load_annotations_from_json
from .utils.image_loader import ImageLoader, ImageLoadError

from src.shi_core.cleaner import Cleaner
from src.shi_core.exceptions import SHIError
from src.shi_core.logging import logger


import os
os.environ["QT_OPENGL"] = "software"          # fuerza render por CPU
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"     # usa llvmpipe si está disponible
os.environ["QT_XCB_GL_INTEGRATION"] = "none"  # evita GLX/EGL (opcional)


class AnnotationsDemoWindow(QMainWindow):
    """Main window for the annotation demo application."""
    
    def __init__(self, img_left, img_right, contrast_left="linear", contrast_right="linear"):
        """Initialize the main window."""
        super().__init__()
        self.setup_window()
        self.setup_ui(img_left, img_right, contrast_left, contrast_right)
        self.setup_menu()
        
    def setup_window(self):
        """Configure window properties."""
        self.setWindowTitle("Two Image Containers — Mirroring Demo")
        self.resize(1400, 800)
        
    def setup_ui(self, img_left, img_right, contrast_left, contrast_right):
        """Set up the UI components."""
        # Create central widget and layout
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)
        
        # Create and configure splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Create image containers
        self.container_a = ImageContainer2D(img_left, contrast_left)
        self.container_b = ImageContainer2D(img_right, contrast_right)
        
        # Add containers to splitter
        splitter.addWidget(self.container_a)
        splitter.addWidget(self.container_b)
        splitter.setSizes([700, 700])
        
        # Set up synchronization
        self.sync_controller = SyncController(self.container_a, self.container_b)
        
        # Create scatter window
        self.scatter_win = ScatterWindow(self.container_a, self.container_b)
        self.scatter_win.show()


    def get_annotations(self):
        """Get all annotations from the left container."""
        return self.container_a.ann_mgr.items()
        
    def clear_annotations(self):
        """Clear all annotations from both containers."""
        self.container_a.ann_mgr.clear()
        self.container_b.ann_mgr.clear()
        
    def get_image_data(self):
        """Get the image data from both containers."""
        return {
            'left': self.container_a.original_image,
            'right': self.container_b.original_image
        }
        
    def save_view(self, filename: str | Path):
        """Save the current view to a file."""
        try:
            self.container_a.view.export_view(str(filename))
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save view: {str(e)}")
            return False



    def setup_menu(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        # Export action
        export_action = QAction("Export annotations...", self)
        export_action.triggered.connect(self._export_annotations)
        file_menu.addAction(export_action)
        
        # Import action
        import_action = QAction("Import annotations...", self)
        import_action.triggered.connect(self._import_annotations)
        file_menu.addAction(import_action)
        
    def _export_annotations(self):
        """Handle export annotations action."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", str(Path.home()), 
            "JSON Files (*.json);;All Files (*.*)"
        )
        if filename:
            try:
                save_annotations_to_json(list(self.container_a.ann_mgr.items()), filename)
                QMessageBox.information(self, "Success", "Annotations exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export annotations: {str(e)}")
                
    def _import_annotations(self):
        """Handle import annotations action."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Annotations", str(Path.home()),
            "JSON Files (*.json);;All Files (*.*)"
        )
        if filename:
            try:
                load_annotations_from_json(filename)
                QMessageBox.information(self, "Success", "Annotations imported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import annotations: {str(e)}")


def parse_arguments() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Morphostructural Analysis Application")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run morphostructural analysis")
    analyze_parser.add_argument(
        "--left",
        type=str,
        help="Path to left image (absorption)"
    )
    analyze_parser.add_argument(
        "--right",
        type=str,
        help="Path to right image (scattering/phase)"
    )
    analyze_parser.add_argument(
        "--contrast",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="Contrast type"
    )
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean temporary files")
    clean_parser.add_argument(
        "--temp",
        action="store_true",
        help="Clean temporary files from morphostructural analysis"
    )
    clean_parser.add_argument(
        "--annotations",
        action="store_true",
        help="Clean saved annotations"
    )
    
    return parser.parse_args()

def load_images(args: Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Load images based on command line arguments."""
    try:
        if args.left and args.right:
            return ImageLoader.load_image_pair(args.left, args.right)
        else:
            logger.info("No image paths provided, using demo images.")
            return ImageLoader.load_demo_images()
    except ImageLoadError as e:
        logger.warning(f"Warning: {str(e)}, using demo images instead.")
        return ImageLoader.load_demo_images()

def clean_files(args: Namespace) -> int:
    """Clean temporary files based on command line arguments."""
    try:
        cleaner = Cleaner()
        
        if args.temp:
            # Clean temporary files from current directory
            logger.info("Cleaning temporary files...")
            cleaner.clean_extra(Path.cwd())
            
        if args.annotations:
            # Clean annotation files
            logger.info("Cleaning annotation files...")
            annotation_dir = Path.home() / "Documents/CXI/CXI-DATA-ANALYSIS/annotations"
            if annotation_dir.exists():
                cleaner._remove_directory(annotation_dir)
            
        return 0
        
    except SHIError as e:
        logger.error(str(e))
        return 1

def main() -> int:
    """Application entry point."""
    args = parse_arguments()
    
    if args.command == "clean":
        return clean_files(args)
        
    elif args.command == "analyze":
        app = QApplication(sys.argv)
        
        try:
            # Load images
            img_left, img_right = load_images(args)
            
            # Create and show main window
            win = AnnotationsDemoWindow(img_left, img_right, args.contrast, args.contrast)
            win.show()
            
            return app.exec()
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Application error: {str(e)}")
            return 1
            
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())




