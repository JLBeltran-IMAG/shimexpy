from setuptools import setup, find_packages
import pathlib


here = pathlib.Path(__file__).parent

def load_requirements(filename):
    path = here / filename
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

install_reqs = load_requirements("requirements.txt")
dev_reqs = load_requirements("requirements-dev.txt")


setup(
    # Package metadata
    name='shimexpy',
    version='0.1.0',
    author='Jorge Luis Beltran Diaz',
    license='Apache License 2.0',

    # Package description
    description='A Python package for spatial harmonics X-ray imaging.',
    long_description=(here / "readme.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',

    # Package structure
    packages=find_packages(),
    install_requires=install_reqs,
    extras_require={
        "dev": dev_reqs
    },
    python_requires='>=3.7',

    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 3 - Alpha', # en desarrollo, no estable
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License', # Apache 2.0
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: POSIX :: Linux', # específicamente Linux
    ],
)

