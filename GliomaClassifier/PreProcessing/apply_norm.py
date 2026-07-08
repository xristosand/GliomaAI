import numpy as np
import SimpleITK as sitk

def z_score_normalize(sitk_image):
    
    """
    Apply z-score normalization to a SimpleITK image.

    Parameters:
    sitk_image (SimpleITK.Image): The input image to be normalized.

    Returns:
    SimpleITK.Image: The z-score normalized image.
    """
    # Convert the SimpleITK image to a NumPy array
    image_array = sitk.GetArrayFromImage(sitk_image)

    mask = image_array > 0
    tissue_array = image_array[mask]

    if tissue_array.size == 0:
        print("No tissue values found in the image for normalization.")
        normalized_array = image_array.astype(np.float32, copy=True)
    else:
        # Calc 0.5th and 99.5th percentiles to exclude outliers
        p05 = np.percentile(tissue_array, 0.5)
        p995 = np.percentile(tissue_array, 99.5)

        # Clip the tissue values to the 0.5th and 99.5th percentiles
        tissue_array_clipped = np.clip(tissue_array, p05, p995)

        # Calculate the mean and standard deviation of the tissue values
        mean_val = tissue_array_clipped.mean()
        std_val = tissue_array_clipped.std()

        # Avoid division by zero
        if std_val == 0:
            std_val = 1

        # Apply z-score normalization only to tissue voxels.
        normalized_array = image_array.astype(np.float32, copy=True)
        normalized_array[mask] = (tissue_array - mean_val) / std_val

    # Convert the normalized array back to a SimpleITK image
    normalized_image = sitk.GetImageFromArray(normalized_array)
    
    # Copy the original image's metadata (spacing, origin, direction)
    normalized_image.SetSpacing(sitk_image.GetSpacing())
    normalized_image.SetOrigin(sitk_image.GetOrigin())
    normalized_image.SetDirection(sitk_image.GetDirection())

    return normalized_image