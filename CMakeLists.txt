cmake_minimum_required(VERSION 3.15.0)
project(TorchEXT LANGUAGES CUDA CXX)

# set flag
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Release)

# find python
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_INTERP python)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")

# find CUDA
find_package(CUDA REQUIRED)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_ARCH)
set(TORCH_CUDA_ARCH_LIST "${INSTALLED_GPU_ARCH}")

# find torch
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
list(APPEND TorchEXT_EXTERNAL_LIB ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# find pybind11
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_pybind11.py FIND_PYBIND_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_PYBIND_PY}
  OUTPUT_VARIABLE PYBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_pybind11.py output: ${PYBIND_CMAKE_DIR}")
set(pybind11_DIR ${PYBIND_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# Define our library target
file(GLOB TorchEXT_SRC
  src/*.cc
  src/*.cu
)

# Set include
pybind11_add_module(TorchEXTLib ${TorchEXT_SRC})
set_target_properties(TorchEXTLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TorchEXTLib PROPERTIES CUDA_STANDARD 17)
set_target_properties(TorchEXTLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
TARGET_COMPILE_OPTIONS(
  TorchEXTLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Enable C++17
target_compile_features(TorchEXTLib PRIVATE cxx_std_17)

# Link against
target_link_libraries(TorchEXTLib PRIVATE ${TorchEXT_EXTERNAL_LIB})