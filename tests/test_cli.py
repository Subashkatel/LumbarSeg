"""
Tests for lumbarseg.cli module.
"""

import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from lumbarseg import cli, __version__


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_required_arguments(self):
        """Test that -i and -o are required."""
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["lumbarseg"]):
                cli.main()

    def test_missing_input(self):
        """Test error when input is missing."""
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["lumbarseg", "-o", "output.nii.gz"]):
                cli.main()

    def test_missing_output(self):
        """Test error when output is missing."""
        with pytest.raises(SystemExit):
            with patch.object(sys, "argv", ["lumbarseg", "-i", "input.nii.gz"]):
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


class TestCLIPreviewAndSplit:
    """Test CLI preview and split options."""

    @patch("lumbarseg.cli.split_segmentation")
    @patch("lumbarseg.cli.segment")
    def test_split_option(self, mock_segment, mock_split, mock_nifti_file, temp_dir):
        """Test --split option calls split_segmentation."""
        output_path = temp_dir / "output.nii.gz"
        mock_split.return_value = {}

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path), "--split"]
        ):
            cli.main()

        mock_segment.assert_called_once()
        mock_split.assert_called_once()

    @patch("lumbarseg.cli.generate_preview")
    @patch("lumbarseg.cli.segment")
    def test_preview_option(self, mock_segment, mock_preview, mock_nifti_file, temp_dir):
        """Test --preview option calls generate_preview."""
        output_path = temp_dir / "output.nii.gz"
        mock_preview.return_value = temp_dir / "preview.png"

        with patch.object(
            sys, "argv",
            ["lumbarseg", "-i", str(mock_nifti_file), "-o", str(output_path), "--preview"]
        ):
            cli.main()

        mock_segment.assert_called_once()
        mock_preview.assert_called_once()
