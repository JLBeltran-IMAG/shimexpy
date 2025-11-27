import numpy as np
import tifffile
from PySide6.QtWidgets import QApplication, QFileDialog
from pathlib import Path


def delete_detector_stripes(image, stripe_rows, stripe_cols):
    """
    Remueve las filas y columnas (stripes) especificadas de la imagen.

    Parameters:
        image (np.ndarray): La imagen de entrada como arreglo de NumPy.
        stripe_rows (list): Lista de índices de filas a eliminar.
        stripe_cols (list): Lista de índices de columnas a eliminar.

    Returns:
        np.ndarray: La imagen con los stripes eliminados.
    """
    # Eliminar las filas especificadas
    image_clean = np.delete(image, stripe_rows, axis=0)

    # Eliminar las columnas especificadas
    image_clean = np.delete(image_clean, stripe_cols, axis=1)
    return image_clean


def correcting_stripes(folder=None):
    """
    Processes TIFF images in a directory to correct detector stripes.
    This function corrects detector artifacts by removing specific rows and columns from TIFF images.
    If no folder is provided, it opens a dialog for folder selection. The function processes all TIFF
    files in the selected folder and its subdirectories.
    Parameters
    ----------
    folder : str or Path, optional
        Path to the directory containing TIFF files. If None, opens a folder selection dialog
        (default is None)
    Returns
    -------
    None
    Notes
    -----
    The function:
    - Searches recursively for .tif files in the given directory
    - Removes predefined detector stripes (specific rows and columns)
    - Overwrites original files with corrected versions
    - Uses specific indices for stripe removal:
        - Rows: [2944, 2945]
        - Columns: [295, 722, 1167, 1388, 1541, 2062, 2302, 2303]
    Requires
    --------
    - tifffile
    - PyQt for folder dialog (if no folder is provided)
    - delete_detector_stripes function
    Example
    -------
    >>> correcting_stripes('/path/to/tiff/files')
    >>> correcting_stripes()  # Opens folder selection dialog
    """
    if folder:
        base_path = Path(folder)
    else:
        app = QApplication([])
        default_folder = Path.home() / "Documents" / "CXI" / "CXI-DATA-ACQUISITION"
        folder = QFileDialog.getExistingDirectory(None, "Elige una carpeta", str(default_folder))

        # Verificar si el usuario canceló la selección
        if not folder:
            return

        app.quit()

        base_path = Path(folder)
    
    # Recopilar los directorios que contienen archivos TIFF
    image_dirs = {path.parent for path in base_path.rglob("*.tif")}

    # Definir los índices de filas y columnas a eliminar
    stripe_rows = [2944, 2945]
    stripe_cols = [295, 722, 1167, 1388, 1541, 2062, 2302, 2303]

    for directory in image_dirs:
        for file_path in directory.glob("*.tif"):
            try:
                # Leer la imagen TIFF
                image = tifffile.imread(file_path)

                # Eliminar los stripes especificados
                image_clean = delete_detector_stripes(image, stripe_rows, stripe_cols)
                
                # Sobrescribir el archivo original con la imagen corregida
                tifffile.imwrite(file_path, image_clean, imagej=True)

            except Exception as e:
                # Keep error message for debugging
                print(f"Error procesando {file_path}: {e}")



