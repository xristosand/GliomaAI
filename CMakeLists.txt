cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(GliomaAI)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/xristosand/GliomaAI")

set(EXTENSION_CATEGORY "Radiology")

set(EXTENSION_CONTRIBUTORS
  "Christos Andrianos (University of West Attica)"
)

set(EXTENSION_DESCRIPTION
  "3D Slicer extension for automated glioma grading from T1CE and FLAIR MRI using deep learning models with integrated explainable AI visualization."
)

# Θα ενημερωθεί όταν το repository γίνει public
set(EXTENSION_ICONURL
  "https://raw.githubusercontent.com/xristosand/GliomaAI/main/GliomaAI.png"
)

set(EXTENSION_SCREENSHOTURLS
  "https://raw.githubusercontent.com/xristosand/GliomaAI/main/docs/screenshots/classification.png"
  "https://raw.githubusercontent.com/xristosand/GliomaAI/main/docs/screenshots/preprocessing.png"
  "https://raw.githubusercontent.com/xristosand/GliomaAI/main/docs/screenshots/results.png"
)

set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(GliomaClassifier)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
