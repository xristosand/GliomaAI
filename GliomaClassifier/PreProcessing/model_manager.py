import os
import slicer
import logging

MODEL_FILES = [
    "proposed_model_final.h5",
    "resnet10_full_torchscript_ep25.pt",
    "resnet50_full_torchscript.pt",
    "densenet121_full_torchscript.pt",
    "synthstrip.1.pt",
]

def get_models_directory():
    """
    Directory where downloaded models are stored.
    """

    modelsDir = os.path.join(
        slicer.app.userSettings().settingsFilePath,
        "..",
        "GliomaAIModels"
    )

    modelsDir = os.path.abspath(modelsDir)

    if not os.path.exists(modelsDir):
        os.makedirs(modelsDir)

    return modelsDir

def print_models_status():

    print("\n----- GliomaAI Models -----")

    for model in MODEL_FILES:

        path = get_model_path(model)

        print(f"{model}")

        print(f"Exists : {os.path.exists(path)}")

        print(path)

        print()

def get_model_path(model_name):
    return os.path.join(get_models_directory(), model_name)


def missing_models():
    missing = []
    for model in MODEL_FILES:
        if not os.path.exists(get_model_path(model)):
            missing.append(model)
    return missing


def models_exist():
    return len(missing_models()) == 0


def download_models():
    raise NotImplementedError


def extract_models():
    raise NotImplementedError