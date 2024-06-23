#include "integrator.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <string>
#include <cmath>
#include <stdexcept>


/// Main execution on GPU device.

namespace kernel
{
    // Check CUDA for errors after GPU execution and throw them.
    __host__ void throw_errors()
    {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::string message = "CUDA Kernel Error: "
                                + std::string(cudaGetErrorString(error));
            throw CudaError(message);
        }
    }
}


/// Host device definitions.

Integrator::Integrator(const HostField& field, Scalar dt, Scalar dx, Scalar dy)
{
}

Integrator::Integrator(Integrator&& other) noexcept
{
}

Field Integrator::get_field() const
{
    return Field::Random(10, 10);
}