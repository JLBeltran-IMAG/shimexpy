"""Command-line interface for SHI package."""
import argparse
import sys
from pathlib import Path

# Local imports
from .config import config
from .processor import SHIProcessor
from .logging import logger
from .exceptions import SHIError


logger.info("Using shimexpy core functionality")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    main_parser = argparse.ArgumentParser(
        prog="SHI",
        description=
            """
            Command line interface for Spatial Harmonic Imaging,
            using shimexpy core functionality.
            """,
        epilog="For more information, visit https://shimexpy.readthedocs.io"
    )

    subparsers = main_parser.add_subparsers(dest="command", required=True)

    # ---------------------
    # Calculate subcommand
    # ---------------------
    parser_shi = subparsers.add_parser(
        "calculate",
        help="Execute the SHI method."
    )
    parser_shi.add_argument(
        "-m", "--mask_period",
        required=True,
        type=int,
        help="Number of projected pixels of the mask-grid."
    )

    # These parameters are optional in automatic mode.
    # The automatic mode processes all subdirectories in the 'sample' directory
    # located in the current working directory.
    parser_shi.add_argument(
        "-i", "--images",
        type=Path,
        help="Path to sample image(s)"
    )
    parser_shi.add_argument(
        "-r", "--reference",
        type=Path,
        help="Path to reference image(s)"
    )
    parser_shi.add_argument(
        "-d", "--dark",
        type=Path,
        help="Path to dark image(s)"
    )
    parser_shi.add_argument(
        "-b", "--bright",
        type=Path,
        help="Path to bright image(s)"
    )

    # Option to apply angle correction after measurements
    parser_shi.add_argument(
        "--angle-after",
        action="store_true",
        help="Apply angle correction after measurements"
    )

    # Option to select phase unwrapping method
    parser_shi.add_argument(
        "--unwrap-phase",
        type=str,
        choices=list(config.UNWRAP_METHODS.keys()),
        help="Select phase unwrapping method"
    )

    parser_shi.add_argument(
        "--allow-crop",
        action="store_true",
        help="Enable cropping of images."
    )

    return main_parser


def run_cli() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.command == "calculate":
            processor = SHIProcessor(
                mask_period=args.mask_period,
                unwrap_method=args.unwrap_phase,
                allow_crop=args.allow_crop
            )

            if not args.images and not args.reference:
                # Handle automatic mode
                measurement_directory = Path.cwd()
                images_path = measurement_directory / "sample"
                dark_path = measurement_directory / "dark"
                reference_path = measurement_directory / "flat"
                bright_path = measurement_directory / "bright"

                # Verify sample directory exists
                if not images_path.exists():
                    raise SHIError(f"Sample directory not found: {images_path}")

                # Process all subdirectories in sample folder
                subdirs = [d for d in images_path.iterdir() if d.is_dir()]
                if not subdirs:
                    raise SHIError(f"No subdirectories found in {images_path}")

                for subdir in subdirs:
                    # Process this subdirectory
                    processor.process_directory(
                        images_path=subdir,
                        reference_path=reference_path,
                        dark_path=dark_path if dark_path.exists() else None,
                        bright_path=bright_path if bright_path.exists() else None,
                        angle_after=args.angle_after
                    )
            else:
                # Handle manual mode
                if not args.images:
                    # If no images are specified, use the default path
                    args.images = Path.cwd() / "sample"

                if not args.images.exists():
                    raise SHIError(f"Images path not found: {args.images}")

                # Check if path contains .tif files or is a directory
                if args.images.is_file() and args.images.suffix.lower() == '.tif':
                    # Single file
                    processor.process_single_image(
                        image_path=args.images,
                        reference_path=args.reference,
                        dark_path=args.dark,
                        bright_path=args.bright,
                        angle_after=args.angle_after
                    )
                elif args.images.is_dir():
                    # Directory containing .tif files
                    tif_files = list(args.images.glob("*.tif"))
                    subdirs = [d for d in args.images.iterdir() if d.is_dir()]

                    if tif_files:
                        # Process .tif files directly in the directory
                        for tif_file in tif_files:
                            processor.process_single_image(
                                image_path=tif_file,
                                reference_path=args.reference,
                                dark_path=args.dark,
                                bright_path=args.bright,
                                angle_after=args.angle_after
                            )
                    elif subdirs:
                        # Process .tif files in subdirectories
                        for subdir in subdirs:
                            subdir_tif_files = list(subdir.glob("*.tif"))
                            if not subdir_tif_files:
                                logger.warning(f"No .tif files found in {subdir}")
                                continue
                            for tif_file in subdir_tif_files:
                                processor.process_single_image(
                                    image_path=tif_file,
                                    reference_path=args.reference,
                                    dark_path=args.dark,
                                    bright_path=args.bright,
                                    angle_after=args.angle_after
                                )
                    else:
                        raise SHIError(f"No .tif files or subdirectories found in {args.images}")
                else:
                    raise SHIError(f"Invalid image path: {args.images}")

        return 0

    except Exception as e:
        logger.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())



