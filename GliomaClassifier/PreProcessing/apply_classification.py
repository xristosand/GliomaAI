import numpy as np
import slicer
import os
import importlib.util
import sys

def ensure_tensorflow_installed():
    import logging

    try:
        import tensorflow as tf
        logging.info(f"TensorFlow already installed: {tf.__version__}")
        return tf

    except ModuleNotFoundError:
        logging.info("TensorFlow not found. Installing...")

        # Recommended CPU version
        slicer.util.pip_install("tensorflow==2.13.0")

        import tensorflow as tf
        logging.info(f"TensorFlow installed successfully: {tf.__version__}")

        return tf


def ensure_torch_installed():
    import logging

    if importlib.util.find_spec("torch") is not None:
        import torch
        logging.info(f"PyTorch available: {torch.__version__}")
        return torch

    try:
        import torch
        logging.info(f"PyTorch available: {torch.__version__}")
        return torch

    except ModuleNotFoundError:
        logging.info("PyTorch not found. Installing...")
        slicer.util.pip_install("typing_extensions>=4.10.0")
        slicer.util.pip_install("torch")
        importlib.invalidate_caches()

        try:
            import torch
            logging.info(f"PyTorch installed successfully: {torch.__version__}")
            return torch
        except Exception as install_error:
            raise RuntimeError(
                "PyTorch installation completed but import still failed. "
                "Please restart Slicer and retry."
            ) from install_error

    except ImportError as e:
        error_text = str(e)

        if "TypeIs" in error_text and "typing_extensions" in error_text:
            logging.info(
                "Detected outdated typing_extensions. Upgrading dependency..."
            )

            slicer.util.pip_install("typing_extensions>=4.10.0")
            importlib.invalidate_caches()

            try:
                import torch
                logging.info(
                    f"PyTorch import recovered: {torch.__version__}"
                )
                return torch
            except Exception as retry_error:
                raise RuntimeError(
                    "PyTorch still cannot be imported after upgrading "
                    "typing_extensions. Please restart Slicer and try again."
                ) from retry_error

        raise RuntimeError(
            f"PyTorch is installed but cannot be imported: {error_text}"
        ) from e


def _prepare_modalities(t1c_array, flair_array):

    # Pad missing modality with zeros.
    if t1c_array is None and flair_array is None:
        raise ValueError("At least one input modality (T1C or FLAIR) is required.")

    if t1c_array is None:
        t1c_array = np.zeros_like(flair_array, dtype=np.float32)

    if flair_array is None:
        flair_array = np.zeros_like(t1c_array, dtype=np.float32)

    if t1c_array.shape != flair_array.shape:
        raise ValueError(
            f"Input modality shape mismatch: T1C={t1c_array.shape}, FLAIR={flair_array.shape}."
        )

    return t1c_array, flair_array


def _infer_expected_channels_tf(model):
    expected_shape = model.input_shape
    if isinstance(expected_shape, list):
        expected_shape = expected_shape[0]

    if expected_shape is None:
        return 2

    if len(expected_shape) > 0 and expected_shape[-1] is not None:
        return int(expected_shape[-1])

    return 2


def _to_binary_probabilities(output_array):
    """Normalize model output to a Keras-like binary probability shape: [1, 2]."""
    arr = np.asarray(output_array, dtype=np.float32).reshape(-1)

    if arr.size == 1:
        value = float(arr[0])
        # If already a probability, keep it. Otherwise interpret as logit.
        if 0.0 <= value <= 1.0:
            p1 = value
        else:
            p1 = 1.0 / (1.0 + np.exp(-value))
        p0 = 1.0 - p1
        return np.array([[p0, p1]], dtype=np.float32)

    if arr.size == 2:
        # Keep valid probability vectors as-is.
        if np.all(arr >= 0.0) and np.isclose(float(arr.sum()), 1.0, atol=1e-3):
            probs = arr
        else:
            # Interpret as logits and apply softmax.
            shifted = arr - np.max(arr)
            exp_scores = np.exp(shifted)
            probs = exp_scores / np.sum(exp_scores)
        return probs[np.newaxis, :].astype(np.float32)

    raise ValueError(
        f"Binary classification expects model output with 1 or 2 values, got {arr.size}."
    )


def _predict_with_tensorflow(t1c_array, flair_array, model_path):
    tf = ensure_tensorflow_installed()
    model = tf.keras.models.load_model(model_path)

    expected_channels = _infer_expected_channels_tf(model)

    if expected_channels == 2:
        input_data = np.stack([t1c_array, flair_array], axis=-1)  # [X,Y,Z,2]
    elif expected_channels == 1:
        input_data = t1c_array[..., np.newaxis]  # [X,Y,Z,1]
    else:
        raise ValueError(f"Unsupported TensorFlow model input channels: {expected_channels}")

    input_data = np.expand_dims(input_data, axis=0)  # [1,X,Y,Z,C]
    raw_output = model.predict(input_data)
    return _to_binary_probabilities(raw_output), model


def _load_pytorch_model(torch, model_path):
    
    
    # Try TorchScript first for maximum compatibility with serialized deployment models.
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        return model
    except Exception:
        pass

    loaded = torch.load(model_path, 
                        map_location="cpu"
                        )

    if isinstance(loaded, torch.nn.Module):
        loaded.eval()
        return loaded

    if isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], torch.nn.Module):
        model = loaded["model"]
        model.eval()
        return model

    raise ValueError(
        "Unsupported PyTorch .pth format. Please provide a TorchScript model or a serialized nn.Module. "
        "Raw state_dict checkpoints need explicit model architecture code before inference."
    )


def _predict_with_pytorch(t1c_array, flair_array, model_path):
    torch = ensure_torch_installed()
    model = _load_pytorch_model(torch, model_path)

    # PyTorch 3D CNN layout: [N,C,D,H,W].
    input_data = np.stack([t1c_array, flair_array], axis=0)  # [2,X,Y,Z]
    input_data = np.expand_dims(input_data, axis=0)  # [1,2,X,Y,Z]
    input_tensor = torch.from_numpy(input_data)

    with torch.no_grad():
        output = model(input_tensor)

    if isinstance(output, (list, tuple)):
        output = output[0]

    output_np = output.detach().cpu().numpy().astype(np.float32)
    return _to_binary_probabilities(output_np), model


def _infer_framework_from_model_path(model_path):
    normalized_path = str(model_path).strip()
    lower_path = normalized_path.lower()
    base_name = os.path.basename(lower_path)

    if lower_path.endswith(".h5") or lower_path.endswith(".keras"):
        return "tensorflow"

    # Support plain .pt/.pth and compound names like *.pth.tar.
    if lower_path.endswith(".pt") or lower_path.endswith(".pth") or ".pt." in base_name or ".pth." in base_name:
        return "pytorch"

    # SavedModel directories typically contain a saved_model.pb file.
    if os.path.isdir(normalized_path) and os.path.exists(os.path.join(normalized_path, "saved_model.pb")):
        return "tensorflow"

    raise ValueError(
        f"Could not infer framework from model path '{model_path}'. "
        "Use a TensorFlow model (.h5/.keras/SavedModel) or PyTorch model (.pt/.pth)."
    )
    

def classify_volume(t1c_array, flair_array, model_path):
    """
    Classify a 3D volume using either TensorFlow or PyTorch.

    Parameters
    ----------
    t1c_array : np.ndarray
        The T1C volume array to classify.
    flair_array : np.ndarray
        The FLAIR volume array to classify.
    model_path : str
        Path to a TensorFlow model (.h5/.keras/SavedModel) or PyTorch model (.pt/.pth).

    Returns
    -------
    tuple : (classification_result, model)
        A tuple containing:
        - classification_result : np.ndarray
            The classification result as a NumPy array.
        - model : TensorFlow or PyTorch model object
            The loaded model object used for classification.
    """

    print("Classification Function called")

    t1c_array, flair_array = _prepare_modalities(t1c_array, flair_array)
    framework = _infer_framework_from_model_path(model_path)
    print(f"Using framework: {framework} | model: {model_path}")

    if framework == "pytorch":
        classification_result, model = _predict_with_pytorch(t1c_array, flair_array, model_path)
        return classification_result, model

    classification_result, model = _predict_with_tensorflow(t1c_array, flair_array, model_path)
    return classification_result, model