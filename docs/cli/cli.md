# Command-line Interface

The `shimexpy_cli` package provides the `shi` command.

## Install from the repository

After [installing the core library](../getting-started/getting-started.md), run
this command from the repository root in the same virtual environment:

```bash
python -m pip install ./shimexpy_cli
```

## Usage

```bash
shi --help
shi calculate --help
```

The `calculate` subcommand requires `--mask_period`: the projected mask period
in pixels. To process a sample with an explicit reference path:

```bash
shi calculate --mask_period 16 --images path/to/sample --reference path/to/flat
```

Replace the paths and example period with values appropriate to your measurement.
Optional flags include `--dark`, `--bright`, `--allow-crop`, `--unwrap-phase`,
and `--angle-after`. Run `shi calculate --help` for their supported values.

[Back to Home](../index.md)
