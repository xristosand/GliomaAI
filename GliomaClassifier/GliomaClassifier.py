import logging
import os
from typing import Annotated, Optional
import numpy as np
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from PreProcessing.apply_skull import apply_skull_stripping
from PreProcessing.apply_n4 import apply_n4_algorithm
from PreProcessing.apply_norm import z_score_normalize
from PreProcessing.prepare_classification import prepare_volume
from PreProcessing.apply_classification import classify_volume
from PreProcessing.apply_XAI import generate_xai_heatmap


#
# GliomaClassifier
#


class GliomaClassifier(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("GliomaClassifier")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#GliomaClassifier">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # GliomaClassifier1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="GliomaClassifier",
        sampleName="GliomaClassifier1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "GliomaClassifier1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="GliomaClassifier1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="GliomaClassifier1",
    )

    # GliomaClassifier2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="GliomaClassifier",
        sampleName="GliomaClassifier2",
        thumbnailFileName=os.path.join(iconsPath, "GliomaClassifier2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="GliomaClassifier2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="GliomaClassifier2",
    )


#
# GliomaClassifierParameterNode
#


@parameterNodeWrapper
class GliomaClassifierParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    outputVolume - The output volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode

    applyRAS: bool = False
    applySkullStrip: bool = False
    applyN4: bool = False
    applyNormalization: bool = False


#
# GliomaClassifierWidget
#


class GliomaClassifierWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._lastHeatmapVolumeNode = None
        self._lastHeatmapReferenceNode = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/GliomaClassifier.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Tab-contained MRML widgets may not receive the scene unless explicitly set.
        for inputSelectorWidget in self._getInputSelectorWidgets():
            inputSelectorWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = GliomaClassifierLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)


        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.applyButton_1.connect("clicked(bool)", self.onApplyButton_1)

        if hasattr(self.ui, "heatmapOpacity"):
            self.ui.heatmapOpacity.connect("valueChanged(double)", self.onHeatmapOpacityChanged)

        # Model selection checkboxes

        self.ui.proposedModelCheckBox.connect(
            "toggled(bool)", lambda checked: self.onModelCheckBoxToggled(self.ui.proposedModelCheckBox, checked))

        self.ui.resnet10CheckBox.connect(
            "toggled(bool)", lambda checked: self.onModelCheckBoxToggled(self.ui.resnet10CheckBox, checked))

        self.ui.resnet50CheckBox.connect(
            "toggled(bool)", lambda checked: self.onModelCheckBoxToggled(self.ui.resnet50CheckBox, checked))
        
        self.ui.densenetCheckBox.connect(
            "toggled(bool)", lambda checked: self.onModelCheckBoxToggled(self.ui.densenetCheckBox, checked))

        # Ensemble checkbox

        self.ui.ensembleCheckBox.connect(
            "toggled(bool)", self.onEnsembleModeChanged)

        # Enforce initial model-selection state based on ensemble checkbox value.
        self.onEnsembleModeChanged(self.ui.ensembleCheckBox.checked)
        

        # Refresh button state when selectors change
        if hasattr(self.ui, "inputSelector"):
            self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)

        if hasattr(self.ui, "inputSelector_1"):
            self.ui.inputSelector_1.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)

        if hasattr(self.ui, "inputSelector_2"):
            self.ui.inputSelector_2.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanApply)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply):
                self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[GliomaClassifierParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply):
                self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        inputSelectorWidget = self._getInputSelectorWidget()
        inputNode = inputSelectorWidget.currentNode() if inputSelectorWidget else None

        if inputNode:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select an input volume node")
            self.ui.applyButton.enabled = False

        # For classification, at least one input is required
        
        t1cNode = None
        flairNode = None

        if hasattr(self.ui, "inputSelector_1"):
            t1cNode = self.ui.inputSelector_1.currentNode()

        if hasattr(self.ui, "inputSelector_2"):
            flairNode = self.ui.inputSelector_2.currentNode()

        if hasattr(self.ui, "applyButton_1"):
            if t1cNode or flairNode:
                self.ui.applyButton_1.enabled = True
                self.ui.applyButton_1.toolTip = _("Run classification")
            else:
                self.ui.applyButton_1.enabled = False
                self.ui.applyButton_1.toolTip = _("Select T1C and/or FLAIR input volume")

    def _getInputSelectorWidget(self):
        inputSelectorWidgets = self._getInputSelectorWidgets()
        return inputSelectorWidgets[0] if inputSelectorWidgets else None

    def _getInputSelectorWidgets(self):
        # Support the staged selector names used by the preprocessing and classification inputs.
        inputSelectorWidgets = []
        for widgetName in ("inputSelector", "inputSelector_1", "inputSelector_2"):
            if hasattr(self.ui, widgetName):
                inputSelectorWidgets.append(getattr(self.ui, widgetName))
        return inputSelectorWidgets


    def _updateHeatmapOverlay(self):
        if self._lastHeatmapVolumeNode is None or self._lastHeatmapReferenceNode is None:
            return

        opacity = self.ui.heatmapOpacity.value/100.0

        layoutManager = slicer.app.layoutManager()

        for viewName in ("Red", "Yellow", "Green"):

            compositeNode = (
                layoutManager
                .sliceWidget(viewName)
                .sliceLogic()
                .GetSliceCompositeNode()
            )

            compositeNode.SetForegroundOpacity(opacity)


    def onHeatmapOpacityChanged(self, value=None) -> None:
        self._updateHeatmapOverlay()
    
    
    # Model selection logic - if ensemble mode is off, only one model can be selected. 
    # If user tries to select a second model, the first one will be automatically deselected.

    def modelCheckBoxes(self):
        return [
            self.ui.proposedModelCheckBox,
            self.ui.resnet10CheckBox,
            self.ui.resnet50CheckBox,
            self.ui.densenetCheckBox
        ]


    def _getXAITargetLayerName(self):
        if self.ui.proposedModelCheckBox.checked:
            return "conv3d_5"

        if self.ui.resnet10CheckBox.checked:
            return "base_model.layer2.0.conv2"

        if self.ui.resnet50CheckBox.checked:
            return "base_model.layer2.3.conv2"

        if self.ui.densenetCheckBox.checked:
            return "features.denseblock2.denselayer6.conv2"

        raise RuntimeError("Select a model before requesting XAI.")


    def onModelCheckBoxToggled(self, senderCheckBox, checked):

        # If ensemble mode is OFF, allow only one selected model
        if not self.ui.ensembleCheckBox.checked:

            if checked:
                for cb in self.modelCheckBoxes():
                    if cb != senderCheckBox:
                        cb.checked = False

            else:
                # Prevent having zero selected models
                if not any(cb.checked for cb in self.modelCheckBoxes()):
                    senderCheckBox.checked = True


    def onEnsembleModeChanged(self, checked):
        # If ensemble mode is disabled, keep only one selected model
        if not checked:
            selected = [cb for cb in self.modelCheckBoxes() if cb.checked]

            if not selected:
                self.ui.proposedModelCheckBox.checked = True
                selected = [self.ui.proposedModelCheckBox]

            first = selected[0]

            for cb in self.modelCheckBoxes():
                cb.checked = (cb == first)

        if hasattr(self.ui, "provideXAIcheckBox"):
            self.ui.provideXAIcheckBox.enabled = not checked
            if checked:
                self.ui.provideXAIcheckBox.checked = False

        if hasattr(self.ui, "renderXAIin3DcheckBox"):
            self.ui.renderXAIin3DcheckBox.enabled = not checked
            if checked:
                self.ui.renderXAIin3DcheckBox.checked = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            inputNode = self.ui.inputSelector.currentNode() if hasattr(self.ui, "inputSelector") else None
            if not inputNode:
                raise RuntimeError("Input node not found.")

            self.logic.process(inputNode,
                               self.ui.applyRASCheckBox.checked,
                               self.ui.applySkullStripCheckBox.checked,
                               self.ui.applyN4CheckBox.checked,
                               self.ui.applyNormalizationCheckBox.checked)
            
    def onApplyButton_1(self) -> None:
        """Run classification when user clicks Apply button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            t1cNode = self.ui.inputSelector_1.currentNode() if hasattr(self.ui, "inputSelector_1") else None
            flairNode = self.ui.inputSelector_2.currentNode() if hasattr(self.ui, "inputSelector_2") else None

            if not t1cNode and not flairNode:
                raise RuntimeError("Select at least one classification input volume.")
            
            t1cArray = None
            t1cPrepInfo = None
            flairArray = None
            flairPrepInfo = None
            
            #Prepare volumes for classification (crop/pad to common shape, then resize to model input size)
            if t1cNode:
                t1cVolume = slicer.util.arrayFromVolume(t1cNode)
                t1cArray, t1cPrepInfo = prepare_volume(t1cVolume, return_metadata=True)

            if flairNode:
                flairVolume = slicer.util.arrayFromVolume(flairNode)
                flairArray, flairPrepInfo = prepare_volume(flairVolume, return_metadata=True)

            
            print("Running classification")
            import time
            s1=time.time()
            
            logging.info(f"T1C input: {t1cNode.GetName() if t1cNode else 'None'}")
            logging.info(f"FLAIR input: {flairNode.GetName() if flairNode else 'None'}")


            selected_models = []

            if self.ui.proposedModelCheckBox.checked:
                selected_models.append({
                    "name": "Proposed Model",
                    "path": self.logic.proposedModelPath
                })

            if self.ui.resnet10CheckBox.checked:
                selected_models.append({
                    "name": "ResNet10",
                    "path": self.logic.resnet10ModelPath
                })

            if self.ui.resnet50CheckBox.checked:

                from PreProcessing import model_manager

                selected_models.append({
                    "name": "ResNet50",
                    "path": model_manager.ensure_resnet50(
                        self.logic.resnet50ModelPath
                    )
                })

            if self.ui.densenetCheckBox.checked:
                selected_models.append({
                    "name": "DenseNet121",
                    "path": self.logic.densenetModelPath
                })

            all_probabilities = []
            loaded_models = []
            model_path = None

            for model_info in selected_models:
                print(f"Running model: {model_info['name']} | {model_info['path']}")

                classification_result, model = classify_volume(
                    t1cArray,
                    flairArray,
                    model_info["path"]
                )

                model_path = model_info["path"]

                probs = np.asarray(classification_result, dtype=np.float32).ravel()

                if probs.shape[0] != 2:
                    raise RuntimeError(
                        f"Expected classification result to have 2 probabilities (LGG and HGG), but got {probs.shape[0]} for model {model_info['name']}.")
                
                all_probabilities.append(probs)
                loaded_models.append({
                    "name": model_info["name"],
                    "model": model
                })

                print(f"Model: {model_info['name']} | Probabilities: {probs}")

            ensemble_mode = self.ui.ensembleCheckBox.checked if hasattr(self.ui, "ensembleCheckBox") else False

            if ensemble_mode:
                probabilities = np.mean(np.stack(all_probabilities, axis=0), axis=0)
                print(f"Ensemble probabilities: {probabilities}")
            else:
                probabilities = all_probabilities[0]
                print(f"Single model probabilities: {probabilities}")


            if probabilities[0] > probabilities[1]:
                predicted_class = "LGG"
            else:
                predicted_class = "HGG"

            self.ui.predictionLabel.setText(predicted_class)

            max_probability = float(np.max(probabilities))
            max_probability = max_probability * 100.0  # Convert to percentage
            self.ui.predictionRate.setText(f"{max_probability:.2f}")
            e1=time.time()
            print("\n")
            print(f"Classification completed in {e1-s1:.2f} seconds")


            if hasattr(self.ui, "provideXAIcheckBox") and self.ui.provideXAIcheckBox.checked:
                # Run Grad-CAM after successful classification and show it as a new scalar volume.
                try:
                    target_layer_name = self._getXAITargetLayerName()

                    heatmap_xyz, pred_index, pred_score = generate_xai_heatmap(
                        t1cArray,
                        flairArray,
                        model,
                        target_layer_name=target_layer_name,
                        model_path=model_path,
                        preprocessing_info=t1cPrepInfo or flairPrepInfo,
                    )

                    displayReferenceNode = t1cNode if t1cNode else flairNode
                    if displayReferenceNode is None:
                        raise RuntimeError("No reference input node available for XAI visualization.")

                    xaiNodeName = f"{displayReferenceNode.GetName()}_GradCAM"

                    xaiNode = slicer.mrmlScene.GetFirstNodeByName(xaiNodeName)
                    if not xaiNode or not xaiNode.IsA("vtkMRMLScalarVolumeNode"):
                        xaiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", xaiNodeName)

                    # Model heatmap is [X,Y,Z]; Slicer expects array order [K,J,I] = [Z,Y,X].
                    heatmap_kji = np.transpose(heatmap_xyz, (2, 1, 0))
                    slicer.util.updateVolumeFromArray(xaiNode, heatmap_kji)
                    xaiNode.CopyOrientation(displayReferenceNode)


                    #slicer.util.setSliceViewerLayers(background=xaiNode, fit=True)

                    self._lastHeatmapVolumeNode = xaiNode
                    self._lastHeatmapReferenceNode = displayReferenceNode
                    initialOpacity = self.ui.heatmapOpacity.value / 100.0
                    slicer.util.setSliceViewerLayers(
                        background=displayReferenceNode,
                        foreground=xaiNode,
                        foregroundOpacity=initialOpacity,
                        fit=True,
                    )
                    

                    # Configure display with a Slicer palette (Rainbow ~= Jet style).
                    if not xaiNode.GetDisplayNode():
                        xaiNode.CreateDefaultDisplayNodes()

                    xaiDisplayNode = xaiNode.GetDisplayNode()
                    if xaiDisplayNode:
                        jetLikeNode = None
                        for nodeName in ("vtkMRMLColorTableNodeFileJet.txt", "Jet"):
                            try:
                                jetLikeNode = slicer.util.getNode(nodeName)
                                break
                            except Exception:
                                continue

                        if jetLikeNode is None:
                            jetLikeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLProceduralColorNode", "Jet")
                            jetColorTransferFunction = vtk.vtkColorTransferFunction()
                            jetColorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.5)
                            jetColorTransferFunction.AddRGBPoint(0.35, 0.0, 0.0, 1.0)
                            jetColorTransferFunction.AddRGBPoint(0.50, 0.0, 1.0, 1.0)
                            jetColorTransferFunction.AddRGBPoint(0.75, 1.0, 1.0, 0.0)
                            jetColorTransferFunction.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
                            jetLikeNode.SetAndObserveColorTransferFunction(jetColorTransferFunction)

                        xaiDisplayNode.SetAndObserveColorNodeID(jetLikeNode.GetID())

                        xaiDisplayNode.SetAutoWindowLevel(False)
                        xaiDisplayNode.SetWindowLevel(1.0, 0.5)
                        
                    logging.info(
                        f"XAI heatmap generated with layer '{target_layer_name}' "
                        f"(class={pred_index}, score={pred_score:.4f})."
                    )
                    print(
                        f"XAI heatmap generated with layer '{target_layer_name}' "
                        f"(class={pred_index}, score={pred_score:.4f})"
                    )

                except Exception as xai_error:
                    logging.warning(f"Classification succeeded, but XAI failed: {xai_error}")
                    print(f"Classification succeeded, but XAI failed: {xai_error}")
                    return  # Do not attempt 3D rendering if heatmap generation failed


            if hasattr(self.ui, "renderXAIin3DcheckBox") and self.ui.renderXAIin3DcheckBox.checked:
                if heatmap_kji is None or xaiNode is None:
                    raise RuntimeError("Enable and run XAI heatmap generation before 3D XAI rendering.")

                # Render the Grad-CAM heatmap itself in 3D instead of converting it
                # to a binary segmentation overlay. Percentiles are estimated from
                # active voxels when possible, because Grad-CAM maps often contain a
                # large near-zero background that compresses useful 3D contrast.
                thresholdPercentile = 92.0
                heatmapMin = float(np.min(heatmap_kji))
                heatmapMax = float(np.max(heatmap_kji))

                if np.isclose(heatmapMin, heatmapMax):
                    heatmapMax = heatmapMin + 1.0

                heatmapRange = heatmapMax - heatmapMin
                activeHeatmapValues = heatmap_kji[heatmap_kji > heatmapMin + heatmapRange * 0.01]
                percentileValues = activeHeatmapValues if activeHeatmapValues.size > 16 else heatmap_kji

                opacityStartValue = float(np.percentile(percentileValues, 55.0))
                contextValue = float(np.percentile(percentileValues, 75.0))
                thresholdValue = float(np.percentile(percentileValues, thresholdPercentile))
                tumorCoreValue = float(np.percentile(percentileValues, 98.0))
                peakValue = float(np.percentile(percentileValues, 99.7))

                opacityStartValue = float(np.clip(opacityStartValue, heatmapMin, heatmapMax))
                contextValue = float(np.clip(contextValue, heatmapMin, heatmapMax))
                thresholdValue = float(np.clip(thresholdValue, heatmapMin, heatmapMax))
                tumorCoreValue = float(np.clip(tumorCoreValue, heatmapMin, heatmapMax))
                peakValue = float(np.clip(peakValue, heatmapMin, heatmapMax))

                volumeRenderingLogic = slicer.modules.volumerendering.logic()
                volumeRenderingDisplayNode = volumeRenderingLogic.GetFirstVolumeRenderingDisplayNode(xaiNode)
                if not volumeRenderingDisplayNode:
                    volumeRenderingLogic.CreateDefaultVolumeRenderingNodes(xaiNode)
                    volumeRenderingDisplayNode = volumeRenderingLogic.GetFirstVolumeRenderingDisplayNode(xaiNode)

                if volumeRenderingDisplayNode:
                    volumeRenderingLogic.UpdateDisplayNodeFromVolumeNode(volumeRenderingDisplayNode, xaiNode)
                    volumeRenderingDisplayNode.SetVisibility(True)

                    volumePropertyNode = volumeRenderingDisplayNode.GetVolumePropertyNode()
                    if volumePropertyNode:
                        volumeProperty = volumePropertyNode.GetVolumeProperty()
                        scalarOpacity = volumeProperty.GetScalarOpacity()
                        rgbTransferFunction = volumeProperty.GetRGBTransferFunction()

                        volumeProperty.SetInterpolationTypeToLinear()
                        volumeProperty.ShadeOn()
                        volumeProperty.SetAmbient(0.35)
                        volumeProperty.SetDiffuse(0.75)
                        volumeProperty.SetSpecular(0.10)
                        volumeProperty.SetSpecularPower(12.0)
                        volumeProperty.SetScalarOpacityUnitDistance(3.0)

                        def addTransferPoints(transferFunction, points, addPoint):
                            lastValue = None
                            minStep = max(heatmapRange * 1e-4, 1e-6)

                            for value, values in points:
                                value = float(np.clip(value, heatmapMin, heatmapMax))
                                if lastValue is not None and value <= lastValue:
                                    value = min(heatmapMax, lastValue + minStep)
                                if lastValue is not None and value <= lastValue:
                                    continue
                                addPoint(transferFunction, value, values)
                                lastValue = value

                        scalarOpacity.RemoveAllPoints()
                        addTransferPoints(
                            scalarOpacity,
                            (
                                (heatmapMin, 0.0),
                                (opacityStartValue, 0.0),
                                (contextValue, 0.012),
                                (thresholdValue, 0.055),
                                (tumorCoreValue, 0.20),
                                (peakValue, 0.34),
                                (heatmapMax, 0.42),
                            ),
                            lambda transferFunction, value, opacity: transferFunction.AddPoint(value, opacity),
                        )

                        rgbTransferFunction.RemoveAllPoints()
                        addTransferPoints(
                            rgbTransferFunction,
                            (
                                (heatmapMin, (0.05, 0.10, 0.18)),
                                (opacityStartValue, (0.05, 0.10, 0.18)),
                                (contextValue, (0.00, 0.70, 1.00)),
                                (thresholdValue, (1.00, 0.92, 0.10)),
                                (tumorCoreValue, (1.00, 0.32, 0.04)),
                                (peakValue, (1.00, 0.04, 0.02)),
                                (heatmapMax, (1.00, 0.00, 0.00)),
                            ),
                            lambda transferFunction, value, color: transferFunction.AddRGBPoint(value, *color),
                        )

                    volumeRenderingDisplayNode.SetCroppingEnabled(False)

                    logging.info(
                        f"3D Grad-CAM volume rendering created using "
                        f"{thresholdPercentile:.1f} percentile as the hotspot opacity knee "
                        f"(threshold={thresholdValue:.4f}, peak={peakValue:.4f})."
                    )
                else:
                    logging.warning("Grad-CAM heatmap was generated, but volume rendering nodes could not be created.")






#
# GliomaClassifierLogic
#


class GliomaClassifierLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

        moduleDir = os.path.dirname(os.path.abspath(__file__))

        self.synthStripModelPath = os.path.join(
            moduleDir,
            "Resources",
            "Models",
            "synthstrip.1.pt"
        )

        logging.info(f"SynthStrip model path: {self.synthStripModelPath}")
        logging.info(f"Model exists: {os.path.exists(self.synthStripModelPath)}")

        
        self.proposedModelPath = os.path.join(
            moduleDir,
            "Resources",
            "Models",
            "proposed_model_final.h5"
        )

        self.resnet10ModelPath = os.path.join(
            moduleDir,
            "Resources",
            "Models",
            "resnet10_full_torchscript_ep25.pt"
        )

        self.resnet50ModelPath = os.path.join(
            moduleDir,
            "Resources",
            "Models",
            "resnet50_full_torchscript.pt"
        )

        self.densenetModelPath = os.path.join(
            moduleDir,
            "Resources",
            "Models",
            "densenet121_full_torchscript.pt"
        )


    def getParameterNode(self):
        return GliomaClassifierParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                applyRAS: bool = False,
                applySkullStrip: bool = False,
                applyN4: bool = False,
                applyNormalization: bool = False) -> vtkMRMLScalarVolumeNode:
        """
        Run the processing algorithm.

        :param inputVolume: input scalar volume
        :param applyRAS: if True, reorient image to RAS
        :param applySkullStrip: if True, apply skull stripping
        :param applyN4: if True, apply N4 bias field correction
        :param applyNormalization: if True, apply normalization
        :return: created output volume node
        """

        if not inputVolume:
            raise ValueError("Input volume is invalid")

        import time
        import SimpleITK as sitk
        import sitkUtils

        startTime = time.time()
        logging.info("Processing started")

        suffix = "_RAS" if applyRAS else "_Copy"

        if applySkullStrip:
            suffix += "_stripped"

        if applyN4:
            suffix += "_N4"

        outputName = inputVolume.GetName() + suffix

        outputVolume = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode",
            outputName
        )

        diffVolume = None
        differenceImage = None

        sitkImage = sitkUtils.PullVolumeFromSlicer(inputVolume)



        # Isotropic Resampling (1mm3) - Default
        # Original info
        orig_spacing = sitkImage.GetSpacing()
        orig_size = sitkImage.GetSize()        
        print("Original size:", orig_size)
        print(
            f"Original spacing: ({orig_spacing[0]:.3f}, {orig_spacing[1]:.3f}, {orig_spacing[2]:.3f})"
        )
        print("\n")
        
        target_spacing = (1.0, 1.0, 1.0)

        if orig_spacing != target_spacing:
            logging.info("Resampling to isotropic 1mm3 spacing")

            new_size = [
                int(round(osz * (ospc / nspc)))
                for osz, ospc, nspc in zip(orig_size, orig_spacing, target_spacing)
            ]

            interpolator = sitk.sitkBSpline

            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolator)
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(sitkImage.GetDirection())
            resampler.SetOutputOrigin(sitkImage.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(0)

            sitkImage = resampler.Execute(sitkImage)
            new_spacing = sitkImage.GetSpacing()
            print("Resampled size:", sitkImage.GetSize())
            print(
                f"Resampled spacing: ({new_spacing[0]:.3f}, {new_spacing[1]:.3f}, {new_spacing[2]:.3f})"
            )
            print("\n")


        if applyRAS:
            logging.info("Applying RAS orientation")
            resultImage = sitk.DICOMOrient(sitkImage, "RAS")
        else:
            logging.info("Skipping RAS orientation")
            resultImage = sitkImage

        if applySkullStrip:
            logging.info("Applying skull stripping")

            resultImage = apply_skull_stripping(
                resultImage,
                model_path=self.synthStripModelPath
            )

            resultImage = sitk.Cast(resultImage, sitk.sitkFloat32)

        else:
            logging.info("Skipping skull stripping")


        if applyN4:

            logging.info("Applying N4 Bias Correction")

            preN4Image = sitk.Cast(resultImage, sitk.sitkFloat32)

            resultImage = apply_n4_algorithm(
                resultImage,
                choice=1)
            resultImage = sitk.Cast(resultImage, sitk.sitkFloat32)

            differenceImage = resultImage - preN4Image

            diffName = outputName + "_Diff"
            diffVolume = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode",
                diffName
            )

            sitkUtils.PushVolumeToSlicer(
                differenceImage,
                diffVolume
            )


        else:

            logging.info("Skipping N4")

        if diffVolume:
            slicer.util.setSliceViewerLayers(
                background=outputVolume,
                foreground=diffVolume,
                foregroundOpacity=0.5)
            
            # Configure difference volume display for better visualization
            diffDisplayNode = diffVolume.GetDisplayNode()

            if diffDisplayNode:
                # Get DivergingBlueRed color table
                divergingNode = slicer.util.getNode("DivergingBlueRed")

                # Apply colormap
                diffDisplayNode.SetAndObserveColorNodeID(
                    divergingNode.GetID()
                )

                # Compute dynamic range
                arr = sitk.GetArrayFromImage(differenceImage)

                vmax = float(arr.max())

                # Avoid zero-range crash
                if vmax == 0:
                    vmax = 1.0

                diffDisplayNode.SetAutoWindowLevel(False)

                diffDisplayNode.SetWindowLevel(
                    vmax,
                    vmax / 2.0
                )

                # Add color bar (legend)
                colorLegend = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(diffVolume)

                if colorLegend:
                    colorLegend.SetTitleText("Bias Difference")
                    colorLegend.SetNumberOfLabels(4)


        else:

            slicer.util.setSliceViewerLayers(background=outputVolume)

        
        if applyNormalization:
            resultImage = z_score_normalize(resultImage)

        sitkUtils.PushVolumeToSlicer(resultImage, outputVolume)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

        return outputVolume


#
# GliomaClassifierTest
#


class GliomaClassifierTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_GliomaClassifier1()

    def test_GliomaClassifier1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("GliomaClassifier1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = GliomaClassifierLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
