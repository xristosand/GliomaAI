import numpy as np
import os

from PreProcessing.apply_classification import ensure_tensorflow_installed, ensure_torch_installed
from PreProcessing.prepare_classification import restore_heatmap_to_original_shape


def _infer_framework_from_model(model):
    if hasattr(model, "predict") and hasattr(model, "get_layer"):
        return "tensorflow"

    torch = ensure_torch_installed()
    if isinstance(model, torch.nn.Module):
        return "pytorch"

    raise ValueError(
        "Unsupported model type for XAI. Expected a TensorFlow Keras model or a PyTorch nn.Module."
    )


def _make_resnet_classifier(torch, base_model, num_classes=2):
    class ResNetClassifier(torch.nn.Module):
        def __init__(self, base_model, num_classes):
            super().__init__()
            self.base_model = base_model
            self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

            last_block = base_model.layer4[-1]
            if hasattr(last_block, "conv3"):
                in_features = last_block.conv3.out_channels
            else:
                in_features = last_block.conv2.out_channels

            self.fc = torch.nn.Linear(in_features, num_classes)

        def forward(self, x):
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    return ResNetClassifier(base_model, num_classes)


def _build_live_pytorch_model_for_xai(model_path, torch):
    if not model_path:
        raise RuntimeError("model_path is required to convert a TorchScript model for Grad-CAM.")

    model_name = os.path.basename(str(model_path)).lower()

    if "resnet10" in model_name:
        architecture_name = "resnet10"
    elif "resnet50" in model_name:
        architecture_name = "resnet50"
    elif "densenet121" in model_name or "densenet" in model_name:
        architecture_name = "densenet121"
    else:
        raise RuntimeError(
            f"No Grad-CAM live-model builder is registered for '{model_name}'. "
            "Add a matching entry in _build_live_pytorch_model_for_xai()."
        )

    if architecture_name == "densenet121":
        from PreProcessing import densenet_model

        return densenet_model.densenet121_3d(in_channels=2, out_channels=2)

    from PreProcessing import resnet_model

    resnet_builder = getattr(resnet_model, architecture_name)
    base_model = resnet_builder(
        sample_input_D=90,
        sample_input_H=120,
        sample_input_W=120,
        num_seg_classes=2,
    )

    # Adapt for 2 input channels (T1C + FLAIR).
    base_model.conv1 = torch.nn.Conv3d(
        2, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False
    )
    return _make_resnet_classifier(torch, base_model, num_classes=2)


def _convert_scriptmodule_for_gradcam(model, model_path, torch):
    live_model = _build_live_pytorch_model_for_xai(model_path, torch)
    live_model.load_state_dict(model.state_dict())
    live_model.eval()
    return live_model


def _prepare_modalities(t1c_array, flair_array):
    t1c_array = None if t1c_array is None else np.asarray(t1c_array, dtype=np.float32)
    flair_array = None if flair_array is None else np.asarray(flair_array, dtype=np.float32)

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


def _to_binary_probabilities(output_array):
    arr = np.asarray(output_array, dtype=np.float32).reshape(-1)

    if arr.size == 1:
        value = float(arr[0])
        if 0.0 <= value <= 1.0:
            p1 = value
        else:
            p1 = 1.0 / (1.0 + np.exp(-value))
        p0 = 1.0 - p1
        return np.array([[p0, p1]], dtype=np.float32)

    if arr.size == 2:
        if np.all(arr >= 0.0) and np.isclose(float(arr.sum()), 1.0, atol=1e-3):
            probs = arr
        else:
            shifted = arr - np.max(arr)
            exp_scores = np.exp(shifted)
            probs = exp_scores / np.sum(exp_scores)
        return probs[np.newaxis, :].astype(np.float32)

    raise ValueError(
        f"Binary classification expects model output with 1 or 2 values, got {arr.size}."
    )

def _build_tensorflow_model_input(t1c_array, flair_array, model):
    expected_shape = model.input_shape
    if isinstance(expected_shape, list):
        expected_shape = expected_shape[0]

    expected_spatial_shape = tuple(expected_shape[1:4]) if len(expected_shape) >= 5 else None
    expected_channels = expected_shape[-1] if len(expected_shape) > 0 else 1

    t1c_array, flair_array = _prepare_modalities(t1c_array, flair_array)

    if expected_spatial_shape is not None:
        if t1c_array.shape != expected_spatial_shape:
            raise ValueError(
                f"T1C volume shape {t1c_array.shape} does not match model spatial shape {expected_spatial_shape}."
            )
        if flair_array.shape != expected_spatial_shape:
            raise ValueError(
                f"FLAIR volume shape {flair_array.shape} does not match model spatial shape {expected_spatial_shape}."
            )

    if expected_channels == 2:
        input_data = np.stack([t1c_array, flair_array], axis=-1)
    elif expected_channels == 1:
        input_data = (t1c_array if t1c_array is not None else flair_array)[..., np.newaxis]
    else:
        raise ValueError(f"Unsupported number of model input channels: {expected_channels}")

    return np.expand_dims(input_data, axis=0).astype(np.float32)


def _build_pytorch_model_input(t1c_array, flair_array):
    t1c_array, flair_array = _prepare_modalities(t1c_array, flair_array)
    input_data = np.stack([t1c_array, flair_array], axis=0)  # [C,X,Y,Z]
    return np.expand_dims(input_data, axis=0).astype(np.float32)  # [1,C,X,Y,Z]


def make_gradcam_heatmap_3d(img_array, model, last_conv_layer_name, pred_index=None):
    tf = ensure_tensorflow_installed()

    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError as exc:
        available_layers = [layer.name for layer in model.layers]
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found in model. Available layers: {available_layers}"
        ) from exc

    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / max_val, tf.zeros_like(heatmap))

    return heatmap.numpy()


def _get_pytorch_target_layer(model, target_layer_name):
    named_modules = dict(model.named_modules())
    if target_layer_name in named_modules:
        return named_modules[target_layer_name]

    alternate_names = []
    if target_layer_name.startswith("base_model."):
        alternate_names.append(target_layer_name[len("base_model."):])
    else:
        alternate_names.append(f"base_model.{target_layer_name}")

    if ".denselayer" in target_layer_name and ".layers." not in target_layer_name:
        alternate_names.append(target_layer_name.replace(".conv", ".layers.conv"))

    for alternate_name in alternate_names:
        if alternate_name in named_modules:
            return named_modules[alternate_name]

    torch = ensure_torch_installed()
    for module_name, module in reversed(list(named_modules.items())):
        if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Conv2d):
            return module

    available_layers = list(named_modules.keys())
    raise ValueError(
        f"Layer '{target_layer_name}' not found in model. Available layers: {available_layers}"
    )


def make_gradcam_heatmap_3d_pytorch(img_tensor, model, last_conv_layer_name, pred_index=None):
    torch = ensure_torch_installed()

    target_layer = _get_pytorch_target_layer(model, last_conv_layer_name)

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        outputs = model(img_tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)

        if pred_index is None:
            pred_index = int(torch.argmax(outputs[0]).item())

        score = outputs[:, pred_index].sum()
        score.backward()

        last_conv_layer_output = activations.get("value")
        grads = gradients.get("value")

        if last_conv_layer_output is None or grads is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        if last_conv_layer_output.ndim != 5:
            raise ValueError(
                f"Expected a 3D convolution layer output with 5 dimensions, got shape {tuple(last_conv_layer_output.shape)}."
            )

        pooled_grads = torch.mean(grads, dim=(0, 2, 3, 4))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = torch.sum(last_conv_layer_output * pooled_grads[:, None, None, None], dim=0)
        heatmap = torch.relu(heatmap)

        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = torch.zeros_like(heatmap)

        return heatmap.detach().cpu().numpy().astype(np.float32), pred_index, float(score.detach().cpu().item())
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _resize_heatmap_to_reference(heatmap_xyz, reference_array):
    from scipy.ndimage import zoom

    target_shape_xyz = tuple(np.asarray(reference_array).shape)

    if heatmap_xyz.shape == target_shape_xyz:
        resized_heatmap = heatmap_xyz.astype(np.float32)
    else:
        zoom_factors = [target / current for target, current in zip(target_shape_xyz, heatmap_xyz.shape)]
        resized_heatmap = zoom(heatmap_xyz, zoom_factors, order=1).astype(np.float32)

    resized_heatmap = resized_heatmap - resized_heatmap.min()
    max_val = float(resized_heatmap.max())
    if max_val > 0:
        resized_heatmap = resized_heatmap / max_val

    return resized_heatmap.astype(np.float32)


def generate_xai_heatmap(
    t1c_array,
    flair_array,
    model,
    target_layer_name,
    model_path=None,
    pred_index=None,
    preprocessing_info=None,
):

    framework = _infer_framework_from_model(model)

    if framework == "tensorflow":
        input_data = _build_tensorflow_model_input(t1c_array, flair_array, model)
        prediction = model.predict(input_data, verbose=0)

        if pred_index is None:
            pred_index = int(np.argmax(prediction[0]))

        heatmap_xyz = make_gradcam_heatmap_3d(
            input_data,
            model,
            last_conv_layer_name=target_layer_name,
            pred_index=pred_index,
        )

        pred_score = float(prediction[0][pred_index])

    else:
        torch = ensure_torch_installed()
        
        # --- ScriptModule → Live Module conversion for Grad-CAM hooks ---
        # Strategy: Same axis convention as TensorFlow path
        # Input: Slicer [Z,Y,X] → Convert to [X,Y,Z] via _build_pytorch_model_input → Model expects [1,C,X,Y,Z]
        if type(model).__name__ == "RecursiveScriptModule" or isinstance(model, torch.jit.ScriptModule):
            try:
                model = _convert_scriptmodule_for_gradcam(model, model_path, torch)
            except Exception as e:
                raise RuntimeError(f"Failed to convert TorchScript model to PyTorch Module for Grad-CAM. Error: {e}")
        else:
            # If not ScriptModule, just set to eval mode
            model.eval()
        # -----------------------------------------------------------------------

        # Build input in [1,C,X,Y,Z] format (consistent with TensorFlow path's [X,Y,Z] convention)
        input_data = _build_pytorch_model_input(t1c_array, flair_array)
        input_tensor = torch.from_numpy(input_data)

        with torch.no_grad():
            prediction = model(input_tensor)
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]

        prediction_np = prediction.detach().cpu().numpy()
        prediction_probs = _to_binary_probabilities(prediction_np)

        if pred_index is None:
            pred_index = int(np.argmax(prediction_probs[0]))

        heatmap_xyz, _, _ = make_gradcam_heatmap_3d_pytorch(
            input_tensor,
            model,
            last_conv_layer_name=target_layer_name,
            pred_index=pred_index,
        )

        pred_score = float(prediction_probs[0][pred_index])

    heatmap_xyz = (heatmap_xyz - heatmap_xyz.min()) / (heatmap_xyz.max() - heatmap_xyz.min() + 1e-8)

    if preprocessing_info is not None:
        heatmap_xyz = restore_heatmap_to_original_shape(heatmap_xyz, preprocessing_info)
    else:
        reference_array = t1c_array if t1c_array is not None else flair_array
        if reference_array is not None:
            heatmap_xyz = _resize_heatmap_to_reference(heatmap_xyz, reference_array)

    return heatmap_xyz.astype(np.float32), pred_index, pred_score