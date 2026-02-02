"""
Tests for lumbarseg.python_api module.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lumbarseg import python_api


class TestDetectDevice:
    """Test detect_device() function."""

    def test_returns_string(self):
        """Test that detect_device returns a string."""
        device = python_api.detect_device()
        assert isinstance(device, str)
        assert device in ["cuda", "cpu"]

    def test_never_returns_mps(self):
        """Test that MPS is never returned (disabled for nnU-Net)."""
        device = python_api.detect_device()
        assert device != "mps"

    @patch("torch.cuda.is_available", return_value=True)
    def test_returns_cuda_when_available(self, mock_cuda):
        """Test that CUDA is returned when available."""
        device = python_api.detect_device()
        assert device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_cpu_when_no_cuda(self, mock_cuda):
        """Test that CPU is returned when CUDA not available."""
        device = python_api.detect_device()
        assert device == "cpu"


class TestValidateInput:
    """Test validate_input() function."""

    def test_valid_nifti_gz(self, mock_nifti_file):
        """Test validation of valid .nii.gz file."""
        result = python_api.validate_input(mock_nifti_file, verbose=False)
        assert result is True

    def test_valid_nifti(self, mock_nifti_file_uncompressed):
        """Test validation of valid .nii file."""
        result = python_api.validate_input(mock_nifti_file_uncompressed, verbose=False)
        assert result is True

    def test_nonexistent_file(self, temp_dir):
        """Test that FileNotFoundError is raised for missing file."""
        fake_path = temp_dir / "nonexistent.nii.gz"
        with pytest.raises(FileNotFoundError):
            python_api.validate_input(fake_path, verbose=False)

    def test_wrong_extension(self, temp_dir):
        """Test that ValueError is raised for wrong extension."""
        wrong_file = temp_dir / "test.txt"
        wrong_file.write_text("not a nifti")
        with pytest.raises(ValueError, match="must be a NIfTI file"):
            python_api.validate_input(wrong_file, verbose=False)

    def test_wrong_extension_jpg(self, temp_dir):
        """Test that ValueError is raised for image files."""
        wrong_file = temp_dir / "test.jpg"
        wrong_file.write_bytes(b"fake image")
        with pytest.raises(ValueError, match="must be a NIfTI file"):
            python_api.validate_input(wrong_file, verbose=False)


class TestSetupNnunetEnvironment:
    """Test setup_nnunet_environment() function."""

    def test_sets_environment_variables(self, mock_lumbarseg_home, clean_env):
        """Test that nnU-Net environment variables are set."""
        python_api.setup_nnunet_environment()

        assert "nnUNet_results" in os.environ
        assert "nnUNet_raw" in os.environ
        assert "nnUNet_preprocessed" in os.environ

    def test_creates_directories(self, mock_lumbarseg_home, clean_env):
        """Test that directories are created."""
        python_api.setup_nnunet_environment()

        raw_dir = Path(os.environ["nnUNet_raw"])
        preprocessed_dir = Path(os.environ["nnUNet_preprocessed"])

        assert raw_dir.exists()
        assert preprocessed_dir.exists()


class TestCheckWeightsExist:
    """Test check_weights_exist() function."""

    def test_returns_false_when_no_weights(self, temp_dir):
        """Test that False is returned when weights don't exist."""
        # Set nnUNet_results to an empty directory
        empty_results = temp_dir / "empty_results"
        empty_results.mkdir()
        os.environ["nnUNet_results"] = str(empty_results)

        # Also set a non-existent path for the default location check
        os.environ["LUMBARSEG_HOME"] = str(temp_dir / "nonexistent_home")

        result = python_api.check_weights_exist()
        assert result is False

        # Cleanup
        if "nnUNet_results" in os.environ:
            del os.environ["nnUNet_results"]
        if "LUMBARSEG_HOME" in os.environ:
            del os.environ["LUMBARSEG_HOME"]

    def test_returns_true_when_weights_exist(self, temp_dir):
        """Test that True is returned when at least one fold exists."""
        # Create a fake checkpoint in the expected structure
        from lumbarseg.config import DATASET_NAME, TRAINER, PLANS, CONFIG
        results_dir = (
            temp_dir / "weights" / DATASET_NAME /
            f"{TRAINER}__{PLANS}__{CONFIG}"
        )
        fold_dir = results_dir / "fold_0"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = fold_dir / "checkpoint_final.pth"
        checkpoint.write_bytes(b"fake checkpoint")

        os.environ["nnUNet_results"] = str(temp_dir / "weights")

        result = python_api.check_weights_exist()
        assert result is True

        # Cleanup
        if "nnUNet_results" in os.environ:
            del os.environ["nnUNet_results"]


class TestPrintFilteredOutput:
    """Test _print_filtered_output() function."""

    def test_filters_citation_block(self, capsys):
        """Test that citation blocks are filtered out."""
        output = """Some useful output
###########################################
Please cite the following if you use nnU-Net:
Citation info here
###########################################
More useful output"""

        python_api._print_filtered_output(output)
        captured = capsys.readouterr()

        assert "useful output" in captured.out
        assert "Please cite" not in captured.out
        assert "Citation info" not in captured.out

    def test_preserves_normal_output(self, capsys):
        """Test that normal output is preserved."""
        output = """Line 1
Line 2
Line 3"""

        python_api._print_filtered_output(output)
        captured = capsys.readouterr()

        assert "Line 1" in captured.out
        assert "Line 2" in captured.out
        assert "Line 3" in captured.out


class TestSegment:
    """Test segment() function."""

    def test_invalid_input_raises_error(self, temp_dir):
        """Test that FileNotFoundError is raised for missing input."""
        fake_input = temp_dir / "nonexistent.nii.gz"
        output = temp_dir / "output.nii.gz"

        with pytest.raises(FileNotFoundError):
            python_api.segment(fake_input, output, verbose=False)

    def test_invalid_extension_raises_error(self, temp_dir):
        """Test that ValueError is raised for wrong extension."""
        wrong_file = temp_dir / "test.txt"
        wrong_file.write_text("not a nifti")
        output = temp_dir / "output.nii.gz"

        with pytest.raises(ValueError, match="must be a NIfTI file"):
            python_api.segment(wrong_file, output, verbose=False)


class TestSegmentBatch:
    """Test segment_batch() function."""

    def test_empty_directory_raises_error(self, mock_empty_dir, temp_dir):
        """Test that FileNotFoundError is raised for empty directory."""
        output_dir = temp_dir / "output"

        with pytest.raises(FileNotFoundError, match="No NIfTI files found"):
            python_api.segment_batch(mock_empty_dir, output_dir, verbose=False)


class TestGeneratePreview:
    """Test generate_preview() function."""

    def test_generates_png(self, mock_nifti_file, mock_segmentation_file, temp_dir):
        """Test that a PNG file is generated."""
        try:
            import matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

        output_path = temp_dir / "preview.png"
        result = python_api.generate_preview(
            mock_nifti_file,
            mock_segmentation_file,
            output_path,
            verbose=False,
        )

        assert result is not None
        assert result.exists()
        assert result.suffix == ".png"

    def test_default_output_path(self, mock_nifti_file, mock_segmentation_file):
        """Test that default output path is used."""
        try:
            import matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

        result = python_api.generate_preview(
            mock_nifti_file,
            mock_segmentation_file,
            verbose=False,
        )

        assert result is not None
        # Default path should be segmentation path with .png extension
        expected_name = mock_segmentation_file.stem.replace(".nii", "") + ".png"
        assert result.name == expected_name


class TestSplitSegmentation:
    """Test split_segmentation() function."""

    def test_creates_separate_masks(self, mock_segmentation_file, temp_dir):
        """Test that separate mask files are created."""
        output_dir = temp_dir / "split"
        output_dir.mkdir()

        result = python_api.split_segmentation(
            mock_segmentation_file,
            output_dir,
            verbose=False,
        )

        assert len(result) == 4
        assert "L_ES" in result
        assert "R_ES" in result
        assert "L_MF" in result
        assert "R_MF" in result

        # Check files exist
        for label_name, path in result.items():
            assert path.exists()

    def test_mask_values_are_binary(self, mock_segmentation_file, temp_dir):
        """Test that output masks are binary (0 and 1 only)."""
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            pytest.skip("nibabel not installed")

        output_dir = temp_dir / "split"
        output_dir.mkdir()

        result = python_api.split_segmentation(
            mock_segmentation_file,
            output_dir,
            verbose=False,
        )

        for label_name, path in result.items():
            img = nib.load(str(path))
            data = img.get_fdata()
            unique_values = np.unique(data)
            assert set(unique_values).issubset({0, 1})


class TestDownloadWithProgress:
    """Test _download_with_progress() function."""

    def test_download_creates_file(self, temp_dir):
        """Test that download creates the destination file."""
        # Use a small, reliable URL for testing
        # Skip if no internet connection
        pytest.importorskip("urllib.request")

        dest = temp_dir / "test_download.txt"

        # Mock the download since we can't guarantee internet access
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "100"
            mock_response.read.side_effect = [b"test content", b""]
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            python_api._download_with_progress(
                "https://example.com/test.txt",
                dest,
                "Testing",
            )

        assert dest.exists()
