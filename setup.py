#!/usr/bin/env python3
"""
LumbarSeg - Setup script for pip installation.

Install with:
    pip install .

Or for development:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="lumbarseg",
    version="1.0.0",
    author="LumbarSeg Team",
    author_email="",
    description="Automatic segmentation of lumbar paraspinal muscles from MRI scans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/USERNAME/LumbarSeg",
    project_urls={
        "Bug Reports": "https://github.com/USERNAME/LumbarSeg/issues",
        "Source": "https://github.com/USERNAME/LumbarSeg",
    },
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "nnunetv2>=2.6",
        "nibabel>=5.0",
        "numpy>=1.24",
        "torch>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "lumbarseg=lumbarseg.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords="medical-imaging segmentation mri lumbar spine nnunet deep-learning",
)
