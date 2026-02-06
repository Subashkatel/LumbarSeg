"""
Tests for lumbarseg.cli module.
"""

import sys
from unittest.mock import patch

import pytest

from lumbarseg import cli, __version__


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_required_arguments(self):
        """Test that -i is required."""
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["lumbarseg"]):
                cli.main()

    def test_version_flag(self, capsys):
        """Test --version shows version."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["lumbarseg", "--version"]):
                cli.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_version_flag_short(self, capsys):
        """Test -v shows version."""
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["lumbarseg", "-v"]):
                cli.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out


class TestCLIOptions:
    """Test CLI option handling."""

    @patch("lumbarseg.cli.segment")
    def test_fast_flag_sets_fold_0(self, mock_segment, mock_nifti_file, temp_dir):
        """Test that --fast sets fold to 0."""
        output_path = temp_dir / "output.nii.gz"

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path), "--fast"]
        ):
            cli.main()

        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args[1]
        assert call_kwargs["fold"] == 0

    @patch("lumbarseg.cli.segment")
    def test_quiet_flag(self, mock_segment, mock_nifti_file, temp_dir):
        """Test that --quiet sets verbose to False."""
        output_path = temp_dir / "output.nii.gz"

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path), "-q"]
        ):
            cli.main()

        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args[1]
        assert call_kwargs["verbose"] is False

    @patch("lumbarseg.cli.segment")
    def test_device_option(self, mock_segment, mock_nifti_file, temp_dir):
        """Test --device option."""
        output_path = temp_dir / "output.nii.gz"

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path), "-d", "cpu"]
        ):
            cli.main()

        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args[1]
        assert call_kwargs["device"] == "cpu"

    @patch("lumbarseg.cli.segment")
    def test_fold_option(self, mock_segment, mock_nifti_file, temp_dir):
        """Test --fold option with specific fold."""
        output_path = temp_dir / "output.nii.gz"

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path), "-f", "2"]
        ):
            cli.main()

        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args[1]
        assert call_kwargs["fold"] == 2

    @patch("lumbarseg.cli.segment")
    def test_fold_all(self, mock_segment, mock_nifti_file, temp_dir):
        """Test --fold all option."""
        output_path = temp_dir / "output.nii.gz"

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path), "-f", "all"]
        ):
            cli.main()

        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args[1]
        assert call_kwargs["fold"] == "all"

    @patch("lumbarseg.cli.segment")
    def test_save_probabilities(self, mock_segment, mock_nifti_file, temp_dir):
        """Test --save-probabilities option."""
        output_path = temp_dir / "output.nii.gz"

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path), "--save-probabilities"]
        ):
            cli.main()

        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args[1]
        assert call_kwargs["save_probabilities"] is True


class TestCLIBatchProcessing:
    """Test CLI batch processing."""

    @patch("lumbarseg.cli.segment_batch")
    def test_directory_input_calls_batch(self, mock_batch, mock_nifti_dir, temp_dir):
        """Test that directory input calls segment_batch."""
        output_dir = temp_dir / "output"

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_dir), "-o", str(output_dir)]
        ):
            cli.main()

        mock_batch.assert_called_once()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_file_not_found_error(self, temp_dir, capsys):
        """Test FileNotFoundError handling."""
        fake_input = temp_dir / "nonexistent.nii.gz"
        output = temp_dir / "output.nii.gz"

        with pytest.raises(SystemExit) as exc_info:
            with patch.object(
                sys, "argv",
                ["lumbarseg", "-i", str(fake_input), "-o", str(output)]
            ):
                cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_value_error(self, temp_dir, capsys):
        """Test ValueError handling for wrong file type."""
        wrong_file = temp_dir / "test.txt"
        wrong_file.write_text("not a nifti")
        output = temp_dir / "output.nii.gz"

        with pytest.raises(SystemExit) as exc_info:
            with patch.object(
                sys, "argv",
                ["lumbarseg", "-i", str(wrong_file), "-o", str(output)]
            ):
                cli.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestCLIGroundTruth:
    """Test CLI ground truth evaluation options."""

    @patch("lumbarseg.cli.segment")
    def test_gt_flag(self, mock_segment, mock_nifti_file, temp_dir):
        """Test --gt flag passes ground truth files to segment."""
        output_path = temp_dir / "output"
        gt1 = temp_dir / "L_ES.nii"
        gt2 = temp_dir / "R_ES.nii"
        gt1.write_bytes(b"fake")
        gt2.write_bytes(b"fake")

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path),
             "--gt", str(gt1), str(gt2)]
        ):
            cli.main()

        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args[1]
        assert call_kwargs["ground_truth"] is not None
        assert len(call_kwargs["ground_truth"]) == 2


class TestCLIAutoNaming:
    """Test automatic output naming."""

    @patch("lumbarseg.cli.segment")
    def test_auto_output_name(self, mock_segment, mock_nifti_file, temp_dir):
        """Test that output is auto-named when -o is not specified."""
        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file)]
        ):
            cli.main()

        mock_segment.assert_called_once()
        call_kwargs = mock_segment.call_args[1]
        output_path = call_kwargs["output"]
        assert str(output_path).endswith("_segmented")
