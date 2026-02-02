"""
Pytest fixtures for LumbarSeg tests.
"""

import os
import tempfile
from pathlib import Path

import pytest
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def mock_nifti_file(temp_dir):
    """Create a mock NIfTI file for testing."""
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    # Create a small 3D volume
    data = np.random.rand(64, 64, 32).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    path = temp_dir / "test_scan.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture
def mock_nifti_file_uncompressed(temp_dir):
    """Create a mock uncompressed NIfTI file."""
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    data = np.random.rand(64, 64, 32).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    path = temp_dir / "test_scan.nii"
    nib.save(img, str(path))
    return path


@pytest.fixture
def mock_segmentation_file(temp_dir):
    """Create a mock multi-label segmentation file."""
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    # Create a segmentation with all 4 labels
    data = np.zeros((64, 64, 32), dtype=np.uint8)

    # Add some labels (L_ES=1, R_ES=2, L_MF=3, R_MF=4)
    data[10:30, 10:25, 10:20] = 1  # L_ES
    data[10:30, 35:50, 10:20] = 2  # R_ES
    data[15:25, 15:22, 12:18] = 3  # L_MF (inside L_ES region)
    data[15:25, 38:45, 12:18] = 4  # R_MF (inside R_ES region)

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)

    path = temp_dir / "test_seg.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture
def mock_empty_dir(temp_dir):
    """Create an empty directory."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()
    return empty_dir


@pytest.fixture
def mock_nifti_dir(temp_dir, mock_nifti_file):
    """Create a directory with multiple NIfTI files."""
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    nifti_dir = temp_dir / "scans"
    nifti_dir.mkdir()

    # Create multiple scan files
    for i in range(3):
        data = np.random.rand(64, 64, 32).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, str(nifti_dir / f"scan_{i:03d}.nii.gz"))

    return nifti_dir


@pytest.fixture
def clean_env():
    """Temporarily clean LumbarSeg-related environment variables."""
    vars_to_clean = [
        "LUMBARSEG_HOME",
        "nnUNet_results",
        "nnUNet_raw",
        "nnUNet_preprocessed",
    ]

    # Save original values
    original = {}
    for var in vars_to_clean:
        original[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_lumbarseg_home(temp_dir, clean_env):
    """Set up a temporary LUMBARSEG_HOME directory."""
    home_dir = temp_dir / ".lumbarseg"
    home_dir.mkdir()
    os.environ["LUMBARSEG_HOME"] = str(home_dir)
    return home_dir
