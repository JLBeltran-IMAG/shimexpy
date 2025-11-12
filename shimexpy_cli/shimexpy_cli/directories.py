import numpy as np
import tifffile as ti
from pathlib import Path


def create_result_directory(
    result_folder: str = "",
    sample_folder: str = ""
) -> Path:
    """
    Create a directory structure for exporting analysis results.

    This function creates a directory structure for exporting analysis results. 
    It creates a main directory based on the provided folder names and a sample folder name.

    Parameters
    ----------
    result_folder : str, optional
        A string specifying the main directory for the results, by default "".
    sample_folder : str, optional
        A string specifying the sample folder name, by default "".

    Returns
    -------
    Path
        Path object representing the main directory for exporting results.
    """
    base_path = Path.home() / "Documents" / "CXI" / "CXI-DATA-ANALYSIS"
    
    # Create the full result path
    if result_folder and sample_folder:
        result_path = base_path / result_folder / sample_folder
    elif result_folder:
        result_path = base_path / result_folder
    else:
        result_path = base_path
    
    # Create the directory and its subdirectories
    result_path.mkdir(parents=True, exist_ok=True)

    return result_path


def create_result_subfolders(
    file_dir: str,
    result_folder: str = "",
    sample_folder: str = ""
) -> tuple[list[Path], Path]:
    """
    Read files from an experiment folder and create subfolders for results.

    This function reads files from a specified directory and creates subfolders for exporting results. 
    It can create a main directory based on the provided folder names and a sample folder name.

    Parameters
    ----------
    file_dir : str
        The path to the directory containing the experiment files.
    result_folder : str, optional
        A string specifying the main directory for the results, by default "".
    sample_folder : str, optional
        A string specifying the sample folder name, by default "".

    Returns
    -------
    tuple
        A tuple containing two elements:
        - A list of Path objects representing the files read from the directory.
        - A Path object representing the main directory for exporting results.
    """
    # Find all .tif files in the specified directory
    path_to_files = [x for x in Path(file_dir).glob("*.tif") if x.is_file()]

    # Create result directory
    if result_folder:
        result_path = create_result_directory(result_folder, sample_folder)
    else:
        result_path = create_result_directory()
    
    return path_to_files, result_path


def create_corrections_folder(path: Path) -> Path:
    """
    Create a folder named "corrections" at the specified location.

    This function creates a folder named "corrections" at the specified location. If the folder already exists, nothing will happen.

    Parameters
    ----------
    path : Path
        Path object representing the location where the "corrections" folder will be created.

    Returns
    -------
    path_to_corrections : Path
        Path object representing the location of the "corrections" folder.

    Notes
    -----
    If the folder already exists, this function does nothing.
    """
    directory_names = [names for names in path.iterdir() if "flat" not in names.name and "results" not in names.name]
    for correction_folders in directory_names:
        path_to_corrections = correction_folders.joinpath("flat_corrections")
        if not path_to_corrections.exists():
            path_to_corrections.mkdir(parents=True, exist_ok=True)

    return path


def export_result_to(
    image_to_save: np.ndarray,
    filename: str,
    path: Path,
    type_of_contrast: str
) -> None:
    """
    Export TIFF files to a default directory.

    This function exports a TIFF image to a default directory specified by the provided path and type of contrast.

    Parameters
    ----------
    image_to_save : ndarray
        The image data to be saved.
    filename : str
        The name of the TIFF file to be saved.
    path : str
        The path to the directory where the TIFF file will be saved.
    type_of_contrast : str
        The type of contrast for the image.

    """
    path_to_save_images = Path(path)

    if not path_to_save_images.exists():
        path_to_save_images.mkdir()

    if type_of_contrast in ["absorption", "scattering", "phase", "phasemap"]:
        if filename:
            path_to_file = path_to_save_images / type_of_contrast / "{}.tif".format(filename)
            ti.imwrite(path_to_file, image_to_save.astype(np.float32), imagej = True)
    # Remove else print statements

