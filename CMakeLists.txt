cmake_minimum_required(VERSION 3.8)
project(activemodelbplus LANGUAGES CXX CUDA)

# Find packages
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Catch2 REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
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
add_library(integrator-kernel SHARED src/integrator.cu src/parameters.cu)
target_include_directories(integrator-kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIRS})
set_target_properties(integrator-kernel PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/activemodelbplus/integrator
)

# Python bindings
pybind11_add_module(integrator src/integrator.cc)
add_dependencies(integrator integrator-kernel)
target_link_libraries(integrator PUBLIC pybind11::headers Eigen3::Eigen PRIVATE integrator-kernel)
target_include_directories(integrator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIRS})
set_target_properties(integrator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/activemodelbplus/integrator)

# Copy Python files to the build directory ready for make install
# Find all Python files recursively
file(GLOB_RECURSE PYTHON_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/activemodelbplus/*.py)

foreach(PY_FILE ${PYTHON_FILES})
  get_filename_component(PY_DIR ${PY_FILE} DIRECTORY)

  add_custom_command(
    TARGET integrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${PY_DIR}
    COMMENT "Creating directory ${PY_DIR} in build directory"
  )

  add_custom_command(
    TARGET integrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${PY_FILE}
            ${CMAKE_CURRENT_BINARY_DIR}/${PY_FILE}
    COMMENT "Copying ${PY_FILE} to build directory"
  )
endforeach()

# Copy link targets into the python directory for local availability
file(GLOB STATIC_OBJECTS ${CMAKE_CURRENT_SOURCE_DIR}/build/activemodelbplus/integrator/*.so)
foreach(SO_FILE ${STATIC_OBJECTS})
  get_filename_component(SO_FILENAME ${SO_FILE} NAME)
  add_custom_command(
    TARGET integrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${SO_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/activemodelbplus/integrator/${SO_FILENAME}
    COMMENT "Copying ${SO_FILE} to module")
endforeach()

# Executable to benchmark aspects of device code.
add_executable(take_derivative benchmark/take_derivative.cu)
target_include_directories(take_derivative PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_custom_command(
  TARGET take_derivative POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/take_derivative
          ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/take_derivative
  COMMAND cuobjdump -res-usage -sass take_derivative >> ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/take_derivative.txt
  COMMENT "Copying take_derivative to benchmark directory"
)


# Unit test executables
add_executable(test_finite_differences test/test_finite_differences.cc)
add_dependencies(test_finite_differences integrator)
target_include_directories(test_finite_differences PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/src
                           ${CMAKE_CURRENT_SOURCE_DIR}/test
                           ${Catch2_INCLUDE_DIRS})

add_executable(test_integrator test/test_integrator.cc)
add_dependencies(test_integrator integrator-kernel test_finite_differences)
target_link_libraries(test_integrator integrator-kernel)
target_include_directories(test_integrator PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/src
                           ${CMAKE_CURRENT_SOURCE_DIR}/test
                           ${Catch2_INCLUDE_DIRS})

# Register unit tests
enable_testing()
add_test(NAME test_finite_differences COMMAND test_finite_differences)
add_test(NAME test_integrator COMMAND test_integrator)

# Run unit tests at end of build process
add_custom_command(
    TARGET test_integrator POST_BUILD
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_integrator test_finite_differences
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unit tests"
)

# Tests to check pybind11 bindings for the python front end
add_test(NAME test_bindings
    COMMAND ${Python_EXECUTABLE} -m pytest
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test
)
add_custom_command(
    TARGET test_integrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}/activemodelbplus ${Python_EXECUTABLE} -m pytest
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test
    COMMENT "Testing python bindings"
)
