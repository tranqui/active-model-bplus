/**
 * This is a simple CUDA kernel for the purposes of inspecting
 * assembly code. After compiling run:
 *   $ cuobjdump -sass take_derivative 
 * to inspect the assembly. To just check resource use:
 *   $ cuobjdump -res-usage take_derivative
 * 
 * If the finite differences coefficients are determined at
 * compile-time, the assembly of `derivative1` and `derivative2`
 * should be identical.
 * 
 * Currently it looks like first derivatives use an unnecessary register
 * because of the zero coefficient which is not optimised away.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include "finite_difference.h"

using namespace finite_difference;

/**
 * This kernel reproduces what we do inside the kernel for `Integrator`.
 * These benchmarks are to confirm that finite difference coefficients are
 * handled at compile-time.
 */
template <Derivative D, std::size_t Order, StaggerGrid Stagger>
__global__ void derivative(Scalar* in, Scalar* out)
{
    *out = apply<D, Order, Stagger>(in);
}

/// Hard-coded kernels that we expect the result above to produce with Order=2.

__global__ void first_derivative_second_order(Scalar* in, Scalar* out)
{
    *out = 0.5 * (in[2] - in[0]);
}

__global__ void second_derivative_second_order(Scalar* in, Scalar* out)
{
    *out = in[2] - 2*in[1] + in[0];
}

int main()
{
    static constexpr int nruns = 100000;
    std::array<Scalar,3> data{0, 1, 2};
    Scalar result;

    Scalar *device_data, *device_result;
    cudaMalloc((void**) &device_data, data.size() * sizeof(Scalar));
    cudaMalloc((void**) &device_result, sizeof(Scalar));
    cudaMemcpy(device_data, data.data(), data.size() * sizeof(Scalar), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int k = 0; k < nruns; ++k) derivative<First,2,Central><<<1,1>>>(device_data, device_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaMemcpy(&result, device_result, sizeof(Scalar), cudaMemcpyDeviceToHost);
    std::cout << "1st derivative (1): " << result << " (execution in " << elapsed << "ms)" << std::endl;

    cudaEventRecord(start);
    for (int k = 0; k < nruns; ++k) first_derivative_second_order<<<1,1>>>(device_data, device_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaMemcpy(&result, device_result, sizeof(Scalar), cudaMemcpyDeviceToHost);
    std::cout << "1st derivative (2): " << result << " (execution in " << elapsed << "ms)" << std::endl;

    cudaEventRecord(start);
    for (int k = 0; k < nruns; ++k) derivative<Second,2,Central><<<1,1>>>(device_data, device_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaMemcpy(&result, device_result, sizeof(Scalar), cudaMemcpyDeviceToHost);
    std::cout << "2nd derivative (1): " << result << " (execution in " << elapsed << "ms)" << std::endl;

    cudaEventRecord(start);
    for (int k = 0; k < nruns; ++k) second_derivative_second_order<<<1,1>>>(device_data, device_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaMemcpy(&result, device_result, sizeof(Scalar), cudaMemcpyDeviceToHost);
    std::cout << "2nd derivative (2): " << result << " (execution in " << elapsed << "ms)" << std::endl;

    cudaFree(device_data);
    cudaFree(device_result);

    return 0;
}
