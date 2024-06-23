#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "integrator.cuh"

namespace py = pybind11;

PYBIND11_MODULE(integrator, m)
{
    py::register_exception<kernel::CudaError>(m, "CudaException");
}