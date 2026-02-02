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
        """Test inference on a single file."""
        output_path = temp_dir / "segmentation.nii.gz"

        result = segment(
            input=real_nifti_file,
            output=output_path,
            fold=0,  # Use single fold for faster testing
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            verbose=True,
        )

        assert result == output_path
        assert output_path.exists()

        # Verify output is a valid NIfTI with correct labels
        import nibabel as nib
        seg = nib.load(str(output_path))
        data = seg.get_fdata()

        # Check shape matches input
        input_img = nib.load(str(real_nifti_file))
        assert data.shape == input_img.shape

        # Check labels are in valid range (0-4)
        unique_labels = np.unique(data)
        assert all(label in [0, 1, 2, 3, 4] for label in unique_labels)

    @pytest.mark.skipif(not weights_available(), reason="Model weights not available")
    def test_inference_with_preview(self, real_nifti_file, temp_dir):
        """Test inference with preview generation."""
        from lumbarseg.python_api import generate_preview

        output_path = temp_dir / "segmentation.nii.gz"

        # Run inference
        segment(
            input=real_nifti_file,
            output=output_path,
            fold=0,
            verbose=False,
        )

        # Generate preview
        preview_path = generate_preview(
            input_image=real_nifti_file,
            segmentation=output_path,
            verbose=False,
        )

        assert preview_path is not None
        assert preview_path.exists()
        assert preview_path.suffix == ".png"

    @pytest.mark.skipif(not weights_available(), reason="Model weights not available")
    def test_inference_with_split(self, real_nifti_file, temp_dir):
        """Test inference with segmentation splitting."""
        from lumbarseg.python_api import split_segmentation

        output_path = temp_dir / "segmentation.nii.gz"

        # Run inference
        segment(
            input=real_nifti_file,
            output=output_path,
            fold=0,
            verbose=False,
        )

        # Split segmentation
        split_dir = temp_dir / "split"
        split_dir.mkdir()
        split_files = split_segmentation(
            segmentation=output_path,
            output_dir=split_dir,
            verbose=False,
        )

        assert len(split_files) == 4
        for label_name in ["L_ES", "R_ES", "L_MF", "R_MF"]:
            assert label_name in split_files
            assert split_files[label_name].exists()


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
        output_path = temp_dir / "segmentation.nii.gz"

        result = segment(
            input=real_nifti_file,
            output=output_path,
            fold=0,
            device="cpu",
            verbose=False,
        )

        assert result == output_path
        assert output_path.exists()


class TestOutputFormats:
    """Test different output format options."""

    @pytest.mark.skipif(not weights_available(), reason="Model weights not available")
    def test_output_is_integer_labels(self, real_nifti_file, temp_dir):
        """Test that output contains integer labels."""
        import nibabel as nib

        output_path = temp_dir / "segmentation.nii.gz"

        segment(
            input=real_nifti_file,
            output=output_path,
            fold=0,
            verbose=False,
        )

        seg = nib.load(str(output_path))
        data = seg.get_fdata()

        # Labels should be integers 0-4
        assert np.allclose(data, data.astype(int))
        assert data.min() >= 0
        assert data.max() <= 4
