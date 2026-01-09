import numpy as np
from pathlib import Path
from shimexpy import save_image


def create_result_directory(
    result_folder: str = "",
    sample_folder: str = ""
) -> Path:
    """
    Create a directory structure for exporting analysis results.
    """
    base_path = Path.home() / "Documents" / "CXI" / "CXI-DATA-ANALYSIS"

    if result_folder and sample_folder:
        result_path = base_path / result_folder / sample_folder
    elif result_folder:
        result_path = base_path / result_folder
    else:
        result_path = base_path

    result_path.mkdir(parents=True, exist_ok=True)
    return result_path


def create_result_subfolders(
    file_dir: str,
    result_folder: str = "",
    sample_folder: str = ""
) -> tuple[list[Path], Path]:
    """
    Read files from an experiment folder and create subfolders for results.
    """
    path_to_files = [x for x in Path(file_dir).glob("*.tif")]

    if result_folder:
        result_path = create_result_directory(result_folder, sample_folder)
    else:
        result_path = create_result_directory()

    return path_to_files, result_path


def create_corrections_folder(path: Path) -> Path:
    """
    Create a folder named "flat_corrections" inside each valid directory.
    """
    directory_names = [
        names for names in path.iterdir()
        if names.is_dir()
        and "flat" not in names.name
        and "results" not in names.name
    ]

    for correction_folders in directory_names:
        path_to_corrections = correction_folders / "flat_corrections"
        path_to_corrections.mkdir(parents=True, exist_ok=True)

    return path


def export_result_to(
    image_to_save: np.ndarray,
    filename: str,
    path: Path,
    type_of_contrast: str
) -> None:
    """
    Export a single TIFF image to the appropriate contrast subfolder.
    """

    path = Path(path)

    # asegurar carpeta base
    path.mkdir(parents=True, exist_ok=True)

    # asegurar subcarpeta por tipo de contraste
    if type_of_contrast in ["absorption", "scattering", "phase", "phasemap"]:
        contrast_dir = path / type_of_contrast
        contrast_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            path_to_file = contrast_dir / f"{filename}.tif"
            save_image(
                image_to_save,
                path_to_file
            )


