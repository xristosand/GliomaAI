import os
import logging
import slicer

RESNET50_URL = (
    "https://github.com/xristosand/GliomaAI/releases/download/v1.0.0/resnet50_full_torchscript.pt"
)


def ensure_resnet50(modelPath):
    """
    Download the ResNet50 model automatically if it is missing.
    """

    if os.path.exists(modelPath):
        logging.info("ResNet50 model already exists.")
        print("ResNet50 model already exists.")
        return modelPath

    os.makedirs(os.path.dirname(modelPath), exist_ok=True)

    logging.info("Downloading ResNet50 model...")
    print("Downloading ResNet50 model...")

    progress = slicer.util.createProgressDialog(
        windowTitle="GliomaAI",
        labelText="Downloading ResNet50 model...\nThis is required only once.",
        maximum=0
    )

    try:

        slicer.util.downloadFile(
            RESNET50_URL,
            modelPath
        )

        logging.info(f"ResNet50 downloaded successfully: {modelPath}")
        print(f"ResNet50 downloaded successfully:\n{modelPath}")

    except Exception as e:

        progress.close()

        slicer.util.errorDisplay(
            f"Unable to download the ResNet50 model.\n\n{e}"
        )

        raise

    finally:

        progress.close()

    return modelPath