"""
Tests for lumbarseg.config module.
"""

import os
from pathlib import Path

import pytest

from lumbarseg import config


class TestConstants:
    """Test that all required constants are defined."""

    def test_dataset_id(self):
        assert config.DATASET_ID == "001"

    def test_dataset_name(self):
        assert config.DATASET_NAME == "Dataset001_LumbarMuscle"

    def test_config_name(self):
        assert config.CONFIG == "3d_fullres"

    def test_trainer(self):
        assert config.TRAINER == "nnUNetTrainerWandb"

    def test_plans(self):
        assert config.PLANS == "nnUNetPlans"

    def test_github_repo(self):
        assert config.GITHUB_REPO == "Subashkatel/LumbarSeg"

    def test_release_tag(self):
        assert config.RELEASE_TAG == "v1.1"

    def test_model_files(self):
        assert len(config.MODEL_FILES) == 5
        for i in range(5):
            assert f"fold_{i}" in config.MODEL_FILES

    def test_label_names(self):
        assert len(config.LABEL_NAMES) == 5
        assert config.LABEL_NAMES[0] == "BG"
        assert config.LABEL_NAMES[1] == "L_ES"
        assert config.LABEL_NAMES[2] == "R_ES"
        assert config.LABEL_NAMES[3] == "L_MF"
        assert config.LABEL_NAMES[4] == "R_MF"


class TestGetLumbarsegHome:
    """Test get_lumbarseg_home() function."""

    def test_default_home(self, clean_env):
        """Test default home is ~/.lumbarseg"""
        home = config.get_lumbarseg_home()
        assert home == Path.home() / ".lumbarseg"
        assert home.exists()

    def test_env_override(self, temp_dir, clean_env):
        """Test LUMBARSEG_HOME environment variable override."""
        custom_home = temp_dir / "custom_lumbarseg"
        os.environ["LUMBARSEG_HOME"] = str(custom_home)

        home = config.get_lumbarseg_home()
        assert home == custom_home
        assert home.exists()

    def test_creates_directory(self, temp_dir, clean_env):
        """Test that the directory is created if it doesn't exist."""
        custom_home = temp_dir / "new_home"
        os.environ["LUMBARSEG_HOME"] = str(custom_home)

        assert not custom_home.exists()
        home = config.get_lumbarseg_home()
        assert home.exists()


class TestGetWeightsDir:
    """Test get_weights_dir() function."""

    def test_returns_weights_subdir(self, mock_lumbarseg_home):
        """Test that weights dir is a subdirectory of home."""
        weights_dir = config.get_weights_dir()
        assert weights_dir == mock_lumbarseg_home / "weights"
        assert weights_dir.exists()

    def test_creates_directory(self, mock_lumbarseg_home):
        """Test that the directory is created."""
        weights_dir = config.get_weights_dir()
        assert weights_dir.is_dir()


class TestGetNnunetResultsDir:
    """Test get_nnunet_results_dir() function."""

    def test_correct_structure(self, mock_lumbarseg_home):
        """Test that the correct nnU-Net directory structure is created."""
        results_dir = config.get_nnunet_results_dir()

        expected_path = (
            mock_lumbarseg_home
            / "weights"
            / config.DATASET_NAME
            / f"{config.TRAINER}__{config.PLANS}__{config.CONFIG}"
        )
        assert results_dir == expected_path
        assert results_dir.exists()

    def test_creates_nested_structure(self, mock_lumbarseg_home):
        """Test that all parent directories are created."""
        results_dir = config.get_nnunet_results_dir()
        assert results_dir.is_dir()

        # Check parent directories exist
        assert results_dir.parent.exists()
        assert results_dir.parent.parent.exists()
