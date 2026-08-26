import os
import logging
import slicer

MODEL_URLS = {
    "ResNet10": "https://github.com/xristosand/GliomaAI/releases/download/v1.0.0/resnet10_full_torchscript_ep25.pt",
    "ResNet50": "https://github.com/xristosand/GliomaAI/releases/download/v1.0.0/resnet50_full_torchscript.pt",
    "DenseNet121": "https://github.com/xristosand/GliomaAI/releases/download/v1.0.0/densenet121_full_torchscript.pt"
}

def ensure_model(model_name, model_path):

    if os.path.exists(model_path):
        logging.info(f"{model_name} already exists.")
        print(f"[GliomaAI] {model_name}: Loaded from local storage.\n")
        return model_path

    url = MODEL_URLS[model_name]

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    progress = slicer.util.createProgressDialog(
        windowTitle="GliomaAI",
        labelText=f"Downloading {model_name}...\nThis is required only once.",
        maximum=0,
    )

    try:

        slicer.util.downloadFile(url, model_path)

        logging.info(f"{model_name} downloaded successfully.")
        print(f"[GliomaAI] {model_name}: Download completed successfully.\n")

    except Exception as e:

        progress.close()

        slicer.util.errorDisplay(
            f"Unable to download {model_name}.\n\n{e}"
        )

        raise

    finally:

        progress.close()

    return model_path