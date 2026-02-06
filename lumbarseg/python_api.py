"""
LumbarSeg - Python API

Simple API for segmenting lumbar paraspinal muscles from MRI scans.

Usage:
    from lumbarseg import segment
    segment("scan.nii.gz", "segmentation.nii.gz")
"""

import os
import platform
import sys
import shutil
import tempfile
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Union, List

from .config import (
    DATASET_ID,
    DATASET_NAME,
    CONFIG,
    TRAINER,
    PLANS,
    LABELS,
    get_lumbarseg_home,
    get_weights_dir,
    get_nnunet_results_dir,
)


def setup_nnunet_environment():
    """
    Set up nnU-Net environment variables.

    This is done automatically - users don't need to set any environment variables.

    Priority:
    1. If nnUNet_results is already set and valid, use it
    2. Check for local nnUNet_results in the package directory
    3. Fall back to ~/.lumbarseg/weights/
    """
    # Check if already set and valid
    if os.environ.get("nnUNet_results"):
        results_path = Path(os.environ["nnUNet_results"])
        model_dir = results_path / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}"
        if model_dir.exists():
            return  # Already configured correctly

    # Check for local nnUNet_results (development/training environment)
    script_dir = Path(__file__).parent.parent.absolute()
    local_results = script_dir / "nnUNet_results"
    local_model_dir = local_results / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}"

    if local_model_dir.exists():
        os.environ["nnUNet_results"] = str(local_results)
        os.environ["nnUNet_raw"] = str(script_dir / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(script_dir / "nnUNet_preprocessed")
        return

    # Fall back to ~/.lumbarseg/weights/
    weights_dir = get_weights_dir()

    # Set nnU-Net environment variables
    os.environ["nnUNet_results"] = str(weights_dir)
    os.environ["nnUNet_raw"] = str(weights_dir / "raw")
    os.environ["nnUNet_preprocessed"] = str(weights_dir / "preprocessed")

    # Create directories
    Path(os.environ["nnUNet_raw"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["nnUNet_preprocessed"]).mkdir(parents=True, exist_ok=True)


def check_weights_exist() -> bool:
    """Check if model weights are installed."""
    # First check the environment variable path
    if os.environ.get("nnUNet_results"):
        results_path = Path(os.environ["nnUNet_results"])
        model_dir = results_path / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}"

        for fold in range(5):
            fold_dir = model_dir / f"fold_{fold}"
            checkpoint = fold_dir / "checkpoint_final.pth"
            if checkpoint.exists():
                return True

    # Check default location
    results_dir = get_nnunet_results_dir()

    # Check for at least one fold
    for fold in range(5):
        fold_dir = results_dir / f"fold_{fold}"
        checkpoint = fold_dir / "checkpoint_final.pth"
        if checkpoint.exists():
            return True

    return False


def _download_with_progress(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    import urllib.request

    # Get file size
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
    except Exception:
        total_size = 0

    # Download with progress
    downloaded = 0
    block_size = 8192

    with urllib.request.urlopen(url) as response:
        with open(dest, 'wb') as out_file:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)

                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    bar_length = 30
                    filled = int(bar_length * downloaded / total_size)
                    bar = '=' * filled + '-' * (bar_length - filled)
                    print(f"\r  {desc}: [{bar}] {percent:5.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

    if total_size > 0:
        print()  # New line after progress bar


def download_weights(verbose: bool = True):
    """
    Download pre-trained model weights from GitHub releases.

    Weights are downloaded to ~/.lumbarseg/weights/
    """
    from .config import GITHUB_REPO, RELEASE_TAG, MODEL_FILES

    if verbose:
        print("Downloading LumbarSeg model weights...")
        print("This only needs to be done once (~1.2 GB total).")

    results_dir = get_nnunet_results_dir()

    # Download each fold
    for fold_name, filename in MODEL_FILES.items():
        fold_dir = results_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = fold_dir / "checkpoint_final.pth"

        if checkpoint_path.exists():
            if verbose:
                print(f"  {fold_name}: Already exists, skipping")
            continue

        url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{filename}"

        try:
            if verbose:
                _download_with_progress(url, checkpoint_path, fold_name)
            else:
                import urllib.request
                urllib.request.urlretrieve(url, checkpoint_path)
        except Exception as e:
            print(f"\nError downloading {fold_name}: {e}")
            print(f"Please download manually from: {url}")
            print(f"And place in: {checkpoint_path}")
            raise

    # Also download plans.json and dataset.json if needed
    _download_config_files(results_dir, verbose)

    if verbose:
        print("Download complete!")


def _download_config_files(results_dir: Path, verbose: bool = True):
    """Download configuration files needed for inference."""
    from .config import GITHUB_REPO, RELEASE_TAG

    config_files = ["plans.json", "dataset.json", "dataset_fingerprint.json"]

    for filename in config_files:
        filepath = results_dir / filename
        if filepath.exists():
            continue

        url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{filename}"

        try:
            import urllib.request
            urllib.request.urlretrieve(url, filepath)
            if verbose:
                print(f"  Downloaded {filename}")
        except Exception:
            # Config files might be embedded in package, ignore download errors
            pass


def validate_input(input_path: Path, verbose: bool = True) -> bool:
    """Validate input file is a valid NIfTI."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = "".join(input_path.suffixes)
    if suffix not in [".nii", ".nii.gz"]:
        raise ValueError(f"Input must be a NIfTI file (.nii or .nii.gz), got: {suffix}")

    # Try to load with nibabel
    try:
        import nibabel as nib
        img = nib.load(str(input_path))
        shape = img.shape
        if verbose:
            print(f"  Input shape: {shape}")
        if len(shape) < 3:
            raise ValueError(f"Expected 3D volume, got {len(shape)}D")
    except ImportError:
        pass  # nibabel not available, skip validation

    return True


def detect_device() -> str:
    """Auto-detect best available device (GPU or CPU).

    Note: MPS (Apple Silicon) is not used because nnU-Net v2 has
    incomplete MPS support and can hang indefinitely. CPU is used
    instead on Mac, which is slower but reliable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Note: MPS disabled - nnU-Net hangs with MPS device
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     return "mps"
    except ImportError:
        pass
    return "cpu"


def segment(
    input: Union[str, Path],
    output: Union[str, Path],
    fold: Union[int, str, List[int]] = "all",
    device: Optional[str] = None,
    verbose: bool = True,
    save_probabilities: bool = False,
    disable_tta: bool = False,
) -> Path:
    """
    Segment lumbar paraspinal muscles from an MRI scan.

    This is the main API function. It handles all setup automatically.

    Parameters
    ----------
    input : str or Path
        Path to input NIfTI file (.nii or .nii.gz)
    output : str or Path
        Path for output segmentation file
    fold : int, str, or list of int
        Which fold(s) to use for inference:
        - "all" or "ensemble": Use all 5 folds (recommended, best accuracy)
        - 0-4: Use a single fold (faster)
        - [0, 1, 2]: Use specific folds
        Default: "all"
    device : str, optional
        Device for inference: "cuda", "cpu", or "mps"
        Auto-detected if not specified.
    verbose : bool
        Print progress messages. Default: True
    save_probabilities : bool
        Save probability maps. Default: False
    disable_tta : bool
        Disable test-time augmentation. ~8x faster but slightly lower accuracy.
        Default: False

    Returns
    -------
    Path
        Path to the output segmentation file

    Examples
    --------
    >>> from lumbarseg import segment
    >>> segment("scan.nii.gz", "segmentation.nii.gz")
    >>> segment("scan.nii.gz", "seg.nii.gz", fold=0, device="cpu")
    """
    input_path = Path(input).absolute()
    output_path = Path(output).absolute()

    total_start = time.time()

    if verbose:
        print("=" * 60)
        print("LumbarSeg - Lumbar Paraspinal Muscle Segmentation")
        print("=" * 60)

    # Validate input
    if verbose:
        print(f"\nInput: {input_path}")
    validate_input(input_path, verbose)

    # Setup environment
    setup_nnunet_environment()

    # Check/download weights
    if not check_weights_exist():
        if verbose:
            print("\nModel weights not found. Downloading...")
        download_start = time.time()
        download_weights(verbose)
        if verbose:
            print(f"  Download time: {time.time() - download_start:.1f}s")

    # Detect device
    if device is None:
        device = detect_device()
    if verbose:
        print(f"Device: {device}")

    # Parse folds
    if fold == "all" or fold == "ensemble":
        folds = [0, 1, 2, 3, 4]
    elif isinstance(fold, int):
        folds = [fold]
    else:
        folds = list(fold)

    if verbose:
        print(f"Folds: {folds}")

    # Run inference
    if verbose:
        print("\nRunning inference...")

    inference_start = time.time()
    _run_nnunet_inference(
        input_path=input_path,
        output_path=output_path,
        folds=folds,
        device=device,
        save_probabilities=save_probabilities,
        verbose=verbose,
        disable_tta=disable_tta,
    )
    inference_time = time.time() - inference_start

    total_time = time.time() - total_start

    if verbose:
        print(f"\nSegmentation saved to: {output_path}")
        print(f"\nTiming:")
        print(f"  Inference: {inference_time:.1f}s")
        print(f"  Total: {total_time:.1f}s")
        print("\nOutput labels:")
        for label_id, label_name in LABELS.items():
            print(f"  {label_id}: {label_name}")

    return output_path


def _print_filtered_output(output: str):
    """Print nnU-Net output, filtering out the citation notice."""
    lines = output.split('\n')
    skip_until_hash = False

    for line in lines:
        # Skip citation block (starts and ends with ####)
        if '######' in line:
            skip_until_hash = not skip_until_hash
            continue
        if skip_until_hash:
            continue
        # Skip empty lines
        if not line.strip():
            continue
        # Print useful output
        print(line)


def _run_spinner(stop_event: threading.Event):
    """Show a spinner with elapsed time while inference is running."""
    chars = "|/-\\"
    start = time.time()
    i = 0
    while not stop_event.is_set():
        elapsed = time.time() - start
        print(f"\r  {chars[i % len(chars)]} Processing... ({elapsed:.0f}s)", end="", flush=True)
        i += 1
        stop_event.wait(0.2)
    elapsed = time.time() - start
    print(f"\r  Inference completed in {elapsed:.1f}s" + " " * 20)


def _run_nnunet_inference(
    input_path: Path,
    output_path: Path,
    folds: List[int],
    device: str,
    save_probabilities: bool,
    verbose: bool,
    disable_tta: bool = False,
):
    """Run nnU-Net inference using subprocess."""
    # Create temp directory for nnU-Net (requires specific file naming)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Prepare input (nnU-Net expects CASEID_0000.nii.gz format)
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        case_id = input_path.stem
        if case_id.endswith(".nii"):
            case_id = case_id[:-4]

        # Create properly named input file
        nnunet_input = input_dir / f"{case_id}_0000.nii.gz"

        if str(input_path).endswith(".nii.gz"):
            shutil.copy2(input_path, nnunet_input)
        else:
            # Need to compress .nii to .nii.gz
            import gzip
            with open(input_path, 'rb') as f_in:
                with gzip.open(nnunet_input, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # Create output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Build nnU-Net command
        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", DATASET_ID,
            "-c", CONFIG,
            "-tr", TRAINER,
            "-f",
        ] + [str(f) for f in folds]

        if device == "cpu":
            cmd.extend(["-device", "cpu"])
        elif device == "mps":
            cmd.extend(["-device", "mps"])

        if save_probabilities:
            cmd.append("--save_probabilities")

        if not verbose:
            cmd.append("--disable_progress_bar")

        if disable_tta:
            cmd.append("--disable_tta")

        # On macOS, disable multiprocessing to prevent deadlocks.
        # Python 3.8+ on macOS uses 'spawn' (not 'fork') for multiprocessing,
        # which causes nnU-Net's preprocessing/export workers to hang.
        # Setting -npp 0 -nps 0 forces sequential mode.
        if platform.system() == "Darwin":
            cmd.extend(["-npp", "0", "-nps", "0"])

        # Run inference with a spinner so the user knows it's working
        try:
            if verbose:
                stop_spinner = threading.Event()
                spinner_thread = threading.Thread(
                    target=_run_spinner, args=(stop_spinner,), daemon=True
                )
                spinner_thread.start()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if verbose:
                stop_spinner.set()
                spinner_thread.join()
                _print_filtered_output(result.stdout)

            if result.returncode != 0:
                if verbose:
                    print(result.stderr)
                raise subprocess.CalledProcessError(result.returncode, cmd)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"nnU-Net inference failed with exit code {e.returncode}")
        except FileNotFoundError:
            raise RuntimeError(
                "nnUNetv2_predict not found. Please install nnU-Net:\n"
                "  pip install nnunetv2"
            )

        # Copy output to final location
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pred_file = output_dir / f"{case_id}.nii.gz"
        if not pred_file.exists():
            # Try to find any output file
            pred_files = list(output_dir.glob("*.nii.gz"))
            if pred_files:
                pred_file = pred_files[0]
            else:
                raise RuntimeError(f"No output file found in {output_dir}")

        shutil.copy2(pred_file, output_path)


def segment_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    fold: Union[int, str, List[int]] = "all",
    device: Optional[str] = None,
    verbose: bool = True,
) -> List[Path]:
    """
    Segment multiple MRI scans in a directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing NIfTI files
    output_dir : str or Path
        Directory for output segmentations
    fold : int, str, or list
        Fold(s) to use. Default: "all"
    device : str, optional
        Device for inference. Auto-detected if not specified.
    verbose : bool
        Print progress messages. Default: True

    Returns
    -------
    List[Path]
        Paths to output segmentation files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find NIfTI files
    nifti_files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))

    if not nifti_files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")

    if verbose:
        print(f"Found {len(nifti_files)} files to process")

    outputs = []
    for i, input_file in enumerate(nifti_files, 1):
        if verbose:
            print(f"\n[{i}/{len(nifti_files)}] {input_file.name}")

        output_name = input_file.name
        if output_name.endswith(".nii"):
            output_name += ".gz"
        output_file = output_dir / output_name

        try:
            segment(
                input=input_file,
                output=output_file,
                fold=fold,
                device=device,
                verbose=verbose,
            )
            outputs.append(output_file)
        except Exception as e:
            print(f"  Error: {e}")

    if verbose:
        print(f"\n\nProcessed {len(outputs)}/{len(nifti_files)} files")

    return outputs


def generate_preview(
    input_image: Union[str, Path],
    segmentation: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    n_slices: int = 3,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Generate a PNG preview image showing the segmentation overlay on the input MRI.

    Parameters
    ----------
    input_image : str or Path
        Path to the input MRI NIfTI file
    segmentation : str or Path
        Path to the segmentation NIfTI file
    output_path : str or Path, optional
        Path for the output PNG. If not specified, uses segmentation path with .png extension.
    n_slices : int
        Number of axial slices to show. Default: 3
    verbose : bool
        Print progress messages. Default: True

    Returns
    -------
    Path or None
        Path to the generated preview image, or None if generation failed
    """
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        if verbose:
            print("Warning: nibabel not available, skipping preview generation")
        return None

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        if verbose:
            print("Warning: matplotlib not available, skipping preview generation")
        return None

    input_image = Path(input_image)
    segmentation = Path(segmentation)

    if output_path is None:
        # Handle .nii.gz extension properly
        seg_name = segmentation.name
        for suffix in ['.nii.gz', '.nii']:
            if seg_name.endswith(suffix):
                seg_name = seg_name[:-len(suffix)]
                break
        output_path = segmentation.parent / f"{seg_name}.png"
    else:
        output_path = Path(output_path)

    if verbose:
        print(f"\nGenerating preview image...")

    # Load images and reorient to RAS+ (standard radiological orientation)
    # This ensures axial slices are along the 3rd dimension regardless of
    # how the NIfTI file was originally stored.
    img = nib.load(str(input_image))
    seg = nib.load(str(segmentation))

    img_ras = nib.as_closest_canonical(img)
    seg_ras = nib.as_closest_canonical(seg)

    img_data = img_ras.get_fdata()
    seg_data = seg_ras.get_fdata()

    # Find axial slices (z-axis in RAS) with most segmentation content
    slice_scores = []
    for z in range(seg_data.shape[2]):
        score = np.sum(seg_data[:, :, z] > 0)
        slice_scores.append((z, score))

    slice_scores.sort(key=lambda x: x[1], reverse=True)
    selected_slices = sorted([s[0] for s in slice_scores[:n_slices]])

    # Color map for segmentation labels
    # 0=transparent, 1=red (L_ES), 2=blue (R_ES), 3=green (L_MF), 4=yellow (R_MF)
    colors = [
        [0, 0, 0, 0],       # 0: Background (transparent)
        [1, 0, 0, 0.5],     # 1: L_ES (red)
        [0, 0, 1, 0.5],     # 2: R_ES (blue)
        [0, 1, 0, 0.5],     # 3: L_MF (green)
        [1, 1, 0, 0.5],     # 4: R_MF (yellow)
    ]
    cmap = ListedColormap(colors)

    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]

    for ax, z in zip(axes, selected_slices):
        # Normalize image slice (axial = [:, :, z] in RAS orientation)
        img_slice = np.rot90(img_data[:, :, z])
        vmin, vmax = np.percentile(img_slice, [1, 99])
        img_normalized = np.clip((img_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # Get segmentation slice
        seg_slice = np.rot90(seg_data[:, :, z])

        # Plot
        ax.imshow(img_normalized, cmap='gray')
        ax.imshow(seg_slice, cmap=cmap, vmin=0, vmax=4)
        ax.set_title(f'Slice {z}')
        ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='L_ES'),
        Patch(facecolor='blue', alpha=0.5, label='R_ES'),
        Patch(facecolor='green', alpha=0.5, label='L_MF'),
        Patch(facecolor='yellow', alpha=0.5, label='R_MF'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.suptitle('LumbarSeg - Segmentation Preview', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def split_segmentation(
    segmentation: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> dict:
    """
    Split a multi-label segmentation into separate binary mask files.

    Parameters
    ----------
    segmentation : str or Path
        Path to the multi-label segmentation NIfTI file
    output_dir : str or Path, optional
        Directory for output files. If not specified, uses same directory as segmentation.
    verbose : bool
        Print progress messages. Default: True

    Returns
    -------
    dict
        Dictionary mapping label names to output file paths
        {
            'L_ES': Path('..._L_ES.nii.gz'),
            'R_ES': Path('..._R_ES.nii.gz'),
            'L_MF': Path('..._L_MF.nii.gz'),
            'R_MF': Path('..._R_MF.nii.gz'),
        }
    """
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        if verbose:
            print("Warning: nibabel not available, skipping split")
        return {}

    from .config import LABEL_NAMES

    segmentation = Path(segmentation)

    if output_dir is None:
        output_dir = segmentation.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSplitting segmentation into separate masks...")

    # Load segmentation
    seg_img = nib.load(str(segmentation))
    seg_data = seg_img.get_fdata()
    affine = seg_img.affine
    header = seg_img.header

    # Get base name (remove .nii.gz)
    base_name = segmentation.name
    for suffix in ['.nii.gz', '.nii']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break

    output_files = {}

    # Create binary mask for each label (skip background = 0)
    for label_id in range(1, 5):
        label_name = LABEL_NAMES.get(label_id, f"label_{label_id}")

        # Create binary mask
        mask = (seg_data == label_id).astype(np.uint8)

        # Create output filename
        output_name = f"{base_name}_{label_name}.nii.gz"
        output_path = output_dir / output_name

        # Save
        mask_img = nib.Nifti1Image(mask, affine, header)
        nib.save(mask_img, str(output_path))

        output_files[label_name] = output_path

        if verbose:
            voxel_count = np.sum(mask)
            print(f"  {label_name}: {voxel_count:,} voxels -> {output_name}")

    return output_files
