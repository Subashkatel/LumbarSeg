#!/usr/bin/env python3
"""
LumbarSeg - Setup script for pip installation.

Install with:
    pip install .

Or for development:
    pip install -e .
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from pathlib import Path


def install_trainer_stub():
    """
    Install nnUNetTrainerWandb stub into nnU-Net package.

    The LumbarSeg model was trained with a custom trainer (nnUNetTrainerWandb).
    For inference, we just need a minimal stub class.
    """
    try:
        import nnunetv2

        nnunet_path = Path(nnunetv2.__path__[0])
        target_dir = nnunet_path / "training" / "nnUNetTrainer" / "variants" / "training_with_wandb"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create trainer file
        trainer_file = target_dir / "nnUNetTrainerWandb.py"
        if not trainer_file.exists():
            trainer_file.write_text('''"""
nnUNetTrainerWandb - Stub for inference compatibility.

The LumbarSeg model was trained with this custom trainer on HPC.
For inference, this stub is sufficient.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWandb(nnUNetTrainer):
    """nnUNetTrainer with W&B support (stub for inference)."""
    pass
''')

        # Create __init__.py
        init_file = target_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("from .nnUNetTrainerWandb import nnUNetTrainerWandb\n")

        print("Installed nnUNetTrainerWandb trainer stub for LumbarSeg")
    except ImportError:
        # nnunetv2 not installed yet, will be installed as dependency
        pass
    except Exception as e:
        print(f"Note: Could not auto-install trainer stub: {e}")
        print("If inference fails, run: ./scripts/install_trainer.sh")


class PostInstallCommand(install):
    """Post-installation: install trainer stub."""
    def run(self):
        install.run(self)
        install_trainer_stub()


class PostDevelopCommand(develop):
    """Post-develop installation: install trainer stub."""
    def run(self):
        develop.run(self)
        install_trainer_stub()


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
    url="https://github.com/Subashkatel/LumbarSeg",
    project_urls={
        "Bug Reports": "https://github.com/Subashkatel/LumbarSeg/issues",
        "Source": "https://github.com/Subashkatel/LumbarSeg",
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
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
)
