#-----------------------------------------------------------------------------
set(MODULE_NAME GliomaClassifier)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py

  PreProcessing/__init__.py
  PreProcessing/apply_classification.py
  PreProcessing/apply_n4.py
  PreProcessing/apply_norm.py
  PreProcessing/apply_skull.py
  PreProcessing/apply_XAI.py
  PreProcessing/densenet_model.py
  PreProcessing/prepare_classification.py
  PreProcessing/resnet_model.py

  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
