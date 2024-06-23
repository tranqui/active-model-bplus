cmake_minimum_required(VERSION 3.8)
project(activemodelbplus LANGUAGES CXX CUDA)

# Find packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

# Set compiler flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_FLAGS "-Ofast")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress 20012 --use_fast_math -ftz=true -prec-div=false -prec-sqrt=false")

# CUDA kernel
add_library(integrator-kernel SHARED src/integrator.cu src/integrator.cuh)
target_include_directories(integrator-kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIRS})
set_target_properties(integrator-kernel PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/activemodelbplus
)

# Python bindings
pybind11_add_module(integrator src/integrator.cc src/integrator.cuh)
target_link_libraries(integrator PUBLIC pybind11::headers Eigen3::Eigen PRIVATE integrator-kernel)
target_include_directories(integrator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIRS})
set_target_properties(integrator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/activemodelbplus)

# Copy Python files to the build directory ready for make install
file(GLOB PYTHON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/activemodelbplus/*.py)
foreach(PY_FILE ${PYTHON_FILES})
  get_filename_component(PY_FILENAME ${PY_FILE} NAME)
  add_custom_command(
    TARGET integrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PY_FILE}
            $<TARGET_FILE_DIR:integrator>/${PY_FILENAME}
    COMMENT "Copying ${PY_FILE} to build directory")
endforeach()

# Copy link targets into the python directory for local availability
file(GLOB STATIC_OBJECTS ${CMAKE_CURRENT_SOURCE_DIR}/build/activemodelbplus/*.so)
foreach(SO_FILE ${STATIC_OBJECTS})
  get_filename_component(SO_FILENAME ${SO_FILE} NAME)
  add_custom_command(
    TARGET integrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${SO_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/activemodelbplus/${SO_FILENAME}
    COMMENT "Copying ${SO_FILE} to module")
endforeach()
