try:
    from scipy.ndimage import zoom
except ImportError:
    raise RuntimeError(
        "SciPy is required but is not available in this Slicer installation."
    )

import numpy as np


def prepare_volume(volume, common_shape=(240,240,180), target_shape=(120,120,90), return_metadata=False):
    
    """Adjust the input `volume` to a common shape then resize for the network.

    Steps:
    1. Crop the volume centrally if any dimension is larger than `common_shape`.
    2. Zero-pad centrally if any dimension is smaller than `common_shape`.
    3. Resize the resulting volume to `target_shape` using spline interpolation.

    Returns a float32 numpy array with shape `target_shape`.
    """

    vol = volume.astype(np.float32)
    vol = np.transpose(vol, (2, 1, 0))  # [Z,Y,X] -> [X,Y,Z] (Slicer to NumPy convention)
    original_shape = vol.shape
    was_prepared = vol.shape != target_shape

    # If the volume is already in the network input shape, do nothing.
    if vol.shape == target_shape:
        print("Volume already in target shape. Skipping preparing!")
        if return_metadata:
            return vol, {
                "original_shape_xyz": original_shape,
                "common_shape": common_shape,
                "target_shape": target_shape,
                "was_prepared": was_prepared,
            }
        return vol

    # 1. Crop (If dims are larger)
    slices = []
    for dim in range(3):
        size = vol.shape[dim]
        target = common_shape[dim]

        if size > target:
            diff = size - target
            start = diff // 2
            end = start + target
            slices.append(slice(start, end))
        else:
            slices.append(slice(None))

    vol = vol[tuple(slices)]

    # 2. Zero Pad (If dims are smaller)
    pad_width = []
    for dim in range(3):
        size = vol.shape[dim]
        target = common_shape[dim]

        if size < target:
            diff = target - size
            before = diff // 2
            after = diff - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))

    if any((b + a) > 0 for (b, a) in pad_width):
        vol = np.pad(vol, pad_width, mode='constant', constant_values=0)

    # 3. Resize to network target
    zoom_factors = (
        float(target_shape[0]) / vol.shape[0],
        float(target_shape[1]) / vol.shape[1],
        float(target_shape[2]) / vol.shape[2],
    )

    resized_vol = zoom(vol, zoom_factors, order=3, prefilter=True)

    print('Volume prepared for classification.')

    if return_metadata:
        return resized_vol, {
            "original_shape_xyz": original_shape,
            "common_shape": common_shape,
            "target_shape": target_shape,
            "was_prepared": was_prepared,
        }

    return resized_vol


def restore_heatmap_to_original_shape(heatmap_xyz, preprocessing_info):
    heatmap_xyz = np.asarray(heatmap_xyz, dtype=np.float32)

    original_shape = tuple(preprocessing_info["original_shape_xyz"])
    common_shape = tuple(preprocessing_info["common_shape"])
    target_shape = tuple(preprocessing_info["target_shape"])
    was_prepared = bool(preprocessing_info.get("was_prepared", True))

    if heatmap_xyz.shape != target_shape:
        zoom_factors = (
            float(target_shape[0]) / heatmap_xyz.shape[0],
            float(target_shape[1]) / heatmap_xyz.shape[1],
            float(target_shape[2]) / heatmap_xyz.shape[2],
        )
        heatmap_xyz = zoom(heatmap_xyz, zoom_factors, order=1, prefilter=True).astype(np.float32)

    if not was_prepared or original_shape == target_shape:
        restored_heatmap = heatmap_xyz
    else:
        common_heatmap = heatmap_xyz
        if common_heatmap.shape != common_shape:
            zoom_factors = (
                float(common_shape[0]) / common_heatmap.shape[0],
                float(common_shape[1]) / common_heatmap.shape[1],
                float(common_shape[2]) / common_heatmap.shape[2],
            )
            common_heatmap = zoom(common_heatmap, zoom_factors, order=1, prefilter=True).astype(np.float32)

        restored_heatmap = np.zeros(original_shape, dtype=np.float32)

        source_slices = []
        target_slices = []
        for original_size, common_size in zip(original_shape, common_shape):
            if original_size > common_size:
                source_slices.append(slice(None))
                start = (original_size - common_size) // 2
                target_slices.append(slice(start, start + common_size))
            elif original_size < common_size:
                start = (common_size - original_size) // 2
                source_slices.append(slice(start, start + original_size))
                target_slices.append(slice(None))
            else:
                source_slices.append(slice(None))
                target_slices.append(slice(None))

        restored_heatmap[tuple(target_slices)] = common_heatmap[tuple(source_slices)]

    restored_heatmap = restored_heatmap - restored_heatmap.min()
    max_val = float(restored_heatmap.max())
    if max_val > 0:
        restored_heatmap = restored_heatmap / max_val

    return restored_heatmap.astype(np.float32)