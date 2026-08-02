cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(effide_ops LANGUAGES CXX CUDA)

if(NOT DEFINED PATH_TO_YOUR_CONDA OR "${PATH_TO_YOUR_CONDA}" STREQUAL "")
    message(WARNING "PATH_TO_YOUR_CONDA is null，please set accordingly")
endif()

if(NOT DEFINED PATH_TO_DEINFER OR "${PATH_TO_DEINFER}" STREQUAL "")
    message(WARNING "PATH_TO_DEINFER is null，please set accordingly")
endif()

set(CMAKE_BUILD_TYPE Release)
set(COMPILE_CC 80) # NOTE: set the cuda arch
set(CMAKE_PREFIX_PATH "${PATH_TO_YOUR_CONDA}/miniconda3/envs/effide/lib/python3.9/site-packages/torch/share/cmake/Torch")


find_package(Torch 1.8 REQUIRED)
find_package(Python3 3.9 REQUIRED Interpreter Development)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
add_library(${PROJECT_NAME} SHARED "${CMAKE_SOURCE_DIR}/cuda_ops.cu")

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES ${COMPILE_CC})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

target_link_libraries(${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})

target_compile_definitions(${PROJECT_NAME} PUBLIC TORCH_EXTENSION_NAME=${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${TORCH_CXX_FLAGS})
target_compile_definitions(${PROJECT_NAME} PUBLIC COMPILE_CC=${COMPILE_CC})

target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC "${PATH_TO_DEINFER}/third_party/wmma_extension/include")
