# # ================================
# # VERY IMPORTANT: LIMIT BLAS/OMP
# # ================================
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ================================
# Standard library imports
# ================================
import argparse
import sys
from pathlib import Path
from multiprocessing import get_context

# ================================
# Local imports
# ================================
from .config import config
from .logging import logger
from .exceptions import SHIError


logger.info("Using shimexpy core functionality")


# ============================================================
# RUN ONE SAMPLE IN AN ISOLATED PROCESS
# ============================================================
def _run_sample_isolated(
    sample_dir: Path,
    reference_path: Path,
    dark_path: Path | None,
    bright_path: Path | None,
    mask_period: int,
    unwrap_phase: str | None,
    allow_crop: bool,
    angle_after: bool,
) -> None:
    """
    Run ONE sample directory in a fresh OS process.
    This guarantees full memory cleanup after completion.
    """
    from .processor import SHIProcessor

    processor = SHIProcessor(
        mask_period=mask_period,
        unwrap_method=unwrap_phase,
        allow_crop=allow_crop,
    )

    processor.process_directory(
        images_path=sample_dir,
        reference_path=reference_path,
        dark_path=dark_path,
        bright_path=bright_path,
        angle_after=angle_after,
    )


# ============================================================
# ARGPARSE
# ============================================================
def create_parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(
        prog="SHI",
        description="Command line interface for Spatial Harmonic Imaging.",
        epilog="https://shimexpy.readthedocs.io",
    )

    subparsers = main_parser.add_subparsers(dest="command", required=True)

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

    parser_shi.add_argument("-i", "--images", type=Path)
    parser_shi.add_argument("-r", "--reference", type=Path)
    parser_shi.add_argument("-d", "--dark", type=Path)
    parser_shi.add_argument("-b", "--bright", type=Path)

    parser_shi.add_argument(
        "--angle-after",
        action="store_true",
        help="Apply angle correction after measurements"
    )

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


# ============================================================
# CLI ENTRYPOINT
# ============================================================
def run_cli() -> int:
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.command != "calculate":
            return 0

        # -----------------------------------------------
        # AUTOMATIC MODE (default measurement structure)
        # -----------------------------------------------
        if not args.images and not args.reference:
            measurement_directory = Path.cwd()
            images_path = measurement_directory / "sample"
            reference_path = measurement_directory / "flat"
            dark_path = measurement_directory / "dark"
            bright_path = measurement_directory / "bright"

            if not images_path.exists():
                raise SHIError(f"Sample directory not found: {images_path}")

            subdirs = [d for d in images_path.iterdir() if d.is_dir()]
            if not subdirs:
                raise SHIError(f"No subdirectories found in {images_path}")

            ctx = get_context("spawn")

            for subdir in sorted(subdirs):
                logger.info(f"Processing sample in isolated process: {subdir.name}")

                proc = ctx.Process(
                    target=_run_sample_isolated,
                    args=(
                        subdir,
                        reference_path,
                        dark_path if dark_path.exists() else None,
                        bright_path if bright_path.exists() else None,
                        args.mask_period,
                        args.unwrap_phase,
                        args.allow_crop,
                        args.angle_after,
                    ),
                )
                proc.start()
                proc.join()

                if proc.exitcode != 0:
                    raise SHIError(f"Processing failed for sample: {subdir}")

        # ----------------------------------------
        # MANUAL MODE (single image or directory)
        # ----------------------------------------
        else:
            from .processor import SHIProcessor

            processor = SHIProcessor(
                mask_period=args.mask_period,
                unwrap_method=args.unwrap_phase,
                allow_crop=args.allow_crop,
            )

            if not args.images:
                args.images = Path.cwd() / "sample"

            if not args.images.exists():
                raise SHIError(f"Images path not found: {args.images}")

            if args.images.is_file() and args.images.suffix.lower() == ".tif":
                processor.process_single_image(
                    image_path=args.images,
                    reference_path=args.reference,
                    dark_path=args.dark,
                    bright_path=args.bright,
                    angle_after=args.angle_after,
                )

            elif args.images.is_dir():
                tif_files = list(args.images.glob("*.tif"))
                subdirs = [d for d in args.images.iterdir() if d.is_dir()]

                if tif_files:
                    for tif_file in tif_files:
                        processor.process_single_image(
                            image_path=tif_file,
                            reference_path=args.reference,
                            dark_path=args.dark,
                            bright_path=args.bright,
                            angle_after=args.angle_after,
                        )

                elif subdirs:
                    for subdir in subdirs:
                        for tif_file in subdir.glob("*.tif"):
                            processor.process_single_image(
                                image_path=tif_file,
                                reference_path=args.reference,
                                dark_path=args.dark,
                                bright_path=args.bright,
                                angle_after=args.angle_after,
                            )
                else:
                    raise SHIError(f"No .tif files or subdirectories in {args.images}")

        return 0

    except Exception as e:
        logger.error(str(e))
        return 1


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    sys.exit(run_cli())
