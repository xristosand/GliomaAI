import os
import sys
import tempfile
import gc
import time
import contextlib
import SimpleITK as sitk

def ensure_synthstrip_installed():
    import slicer

    try:
        from nipreps.synthstrip.cli import main as synthstrip_main
        return synthstrip_main
    except ModuleNotFoundError:
        slicer.util.pip_install("nipreps-synthstrip")
        from nipreps.synthstrip.cli import main as synthstrip_main
        return synthstrip_main


def apply_skull_stripping(input_sitk_image, model_path, verbose=False):
    
    """
    Apply SynthStrip using temporary NIfTI files.

    Parameters
    ----------
    input_sitk_image : SimpleITK.Image
    model_path : str

    Returns
    -------
    stripped_sitk_image : SimpleITK.Image
    mask_sitk_image : SimpleITK.Image
    """

    print("Skull-Stripping Function called")

    synthstrip_main = ensure_synthstrip_installed()
    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:

        input_path = os.path.join(tmpdir, "input.nii.gz")
        stripped_path = os.path.join(tmpdir, "stripped.nii.gz")
        mask_path = os.path.join(tmpdir, "mask.nii.gz")

        # Write temporary input
        sitk.WriteImage(input_sitk_image, input_path)

        # Run SynthStrip, suppressing CLI output unless verbose is requested.
        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "synthstrip",
                "--image", input_path,
                "--out", stripped_path,
                "--mask", mask_path,
                "--model", model_path,
            ]

            if verbose:
                synthstrip_main()
            else:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        synthstrip_main()
        finally:
            sys.argv = original_argv

        # Load only stripped result
        stripped_sitk_image = sitk.ReadImage(stripped_path)

        # Cleanup Python refs before leaving tempdir
        del input_path, stripped_path, mask_path
        gc.collect()

        elapsed = time.time() - start_time
        print(f"     Execution time: {elapsed:.2f} sec")

        return stripped_sitk_image