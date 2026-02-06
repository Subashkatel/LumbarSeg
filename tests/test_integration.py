"""
Integration tests for LumbarSeg.

These tests require model weights to be available and may take longer to run.
Skip with: pytest -m "not integration"
"""

import os
from pathlib import Path

import pytest
import numpy as np

from lumbarseg import segment
from lumbarseg.python_api import check_weights_exist, setup_nnunet_environment


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def weights_available():
    """Check if model weights are available for testing."""
    setup_nnunet_environment()
    return check_weights_exist()


@pytest.fixture
def real_nifti_file():
    """Get a real NIfTI file for integration testing."""
    # Check for test data in the nnUNet_raw directory
    test_file = Path("/scratch/gpfs/MARTONOSI/sk2415/ml/nnUNet_raw/Dataset001_LumbarMuscle/imagesTr/SBT001_0000.nii.gz")
    if test_file.exists():
        return test_file

    # Try to find any NIfTI file in the project
    project_root = Path(__file__).parent.parent
    nifti_files = list(project_root.glob("**/imagesTr/*.nii.gz"))
    if nifti_files:
        return nifti_files[0]

    pytest.skip("No real NIfTI files available for integration testing")


class TestFullInferencePipeline:
    """Test the full inference pipeline."""

    @pytest.mark.skipif(not weights_available(), reason="Model weights not available")
    def test_single_file_inference(self, real_nifti_file, temp_dir):
        """Test inference on a single file produces output folder."""
        output_dir = temp_dir / "results"

        result = segment(
            input=real_nifti_file,
            output=output_dir,
            fold=0,  # Use single fold for faster testing
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            verbose=True,
        )

        assert result == output_dir
        assert output_dir.exists()
        assert (output_dir / "segmentation.nii.gz").exists()
        assert (output_dir / "L_ES.nii.gz").exists()
        assert (output_dir / "R_ES.nii.gz").exists()
        assert (output_dir / "L_MF.nii.gz").exists()
        assert (output_dir / "R_MF.nii.gz").exists()
        assert (output_dir / "preview.png").exists()

        # Verify segmentation is a valid NIfTI with correct labels
        import nibabel as nib
        seg = nib.load(str(output_dir / "segmentation.nii.gz"))
        data = seg.get_fdata()

        # Check labels are in valid range (0-4)
        unique_labels = np.unique(data)
        assert all(label in [0, 1, 2, 3, 4] for label in unique_labels)

    @pytest.mark.skipif(not weights_available(), reason="Model weights not available")
    def test_binary_masks_are_valid(self, real_nifti_file, temp_dir):
        """Test that output binary masks are binary (0 and 1 only)."""
        import nibabel as nib

        output_dir = temp_dir / "results"

        segment(
            input=real_nifti_file,
            output=output_dir,
            fold=0,
            verbose=False,
        )

        for label_name in ["L_ES", "R_ES", "L_MF", "R_MF"]:
            mask_path = output_dir / f"{label_name}.nii.gz"
            assert mask_path.exists()
            data = nib.load(str(mask_path)).get_fdata()
            unique_values = np.unique(data)
            assert set(unique_values).issubset({0, 1})


class TestEnvironmentSetup:
    """Test environment setup for nnU-Net."""

    def test_setup_creates_directories(self, mock_lumbarseg_home, clean_env):
        """Test that setup creates necessary directories."""
        setup_nnunet_environment()

        assert "nnUNet_results" in os.environ
        assert "nnUNet_raw" in os.environ
        assert "nnUNet_preprocessed" in os.environ

        assert Path(os.environ["nnUNet_raw"]).exists()
        assert Path(os.environ["nnUNet_preprocessed"]).exists()


class TestDeviceHandling:
    """Test device handling in inference."""

    @pytest.mark.skipif(not weights_available(), reason="Model weights not available")
    def test_cpu_inference(self, real_nifti_file, temp_dir):
        """Test inference on CPU."""
        output_dir = temp_dir / "results"

        result = segment(
            input=real_nifti_file,
            output=output_dir,
            fold=0,
            device="cpu",
            verbose=False,
        )

        assert result == output_dir
        assert output_dir.exists()
        assert (output_dir / "segmentation.nii.gz").exists()


class TestOutputFormats:
    """Test different output format options."""

    @pytest.mark.skipif(not weights_available(), reason="Model weights not available")
    def test_output_is_integer_labels(self, real_nifti_file, temp_dir):
        """Test that output contains integer labels."""
        import nibabel as nib

        output_dir = temp_dir / "results"

        segment(
            input=real_nifti_file,
            output=output_dir,
            fold=0,
            verbose=False,
        )

        seg = nib.load(str(output_dir / "segmentation.nii.gz"))
        data = seg.get_fdata()

        # Labels should be integers 0-4
        assert np.allclose(data, data.astype(int))
        assert data.min() >= 0
        assert data.max() <= 4
