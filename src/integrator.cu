#include "integrator.h"
#include "parameters.cuh"
#include "for_each.h"
#include "finite_difference.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <string>
#include <cmath>
#include <stdexcept>
#include <random>


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

    /// Physical calculations

    __device__ inline Scalar bulk_chemical_potential(Scalar field)
    {
        return model.a * field + model.b * field * field + model.c * field * field * field;
    }

    __device__ inline Scalar deriv_bulk_chemical_potential(Scalar field)
    {
        return model.a + 2 * model.b * field + 3 * model.c * field * field;
    }

    /// Kernel to determine current and chemical potential.

    // Note this calculation determines $\mu$ and the non-conservative current separately
    __global__ void calculate_current(DeviceField field,
                                      DeviceField chemical_potential,
                                      DeviceCurrent circulating_current,
                                      curandState *random_state)
    {
        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load tile into shared memory.

        __shared__ Scalar tile[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];
        tile[i][j] = field[index];

        // Fill in ghost points.

        if (threadIdx.y < num_ghost)
        {
            tile[i - num_ghost][j] = field[col + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i + tile_rows][j] = field[col + ((row + tile_rows) % nrows) * ncols];
        }

        if (threadIdx.x < num_ghost)
        {
            tile[i][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + row * ncols];
            tile[i][j + tile_cols] = field[(col + tile_cols) % ncols         + row * ncols];
        }

        if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
        {
            tile[i - num_ghost][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i - num_ghost][j + tile_cols] = field[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
            tile[i + tile_rows][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
            tile[i + tile_rows][j + tile_cols] = field[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
        }

        __syncthreads();

        // Surface terms involve derivatives of the field.

        Scalar lap = laplacian(tile, i, j);
        chemical_potential[index] = bulk_chemical_potential(tile[i][j])
                                    - model.kappa * lap
                                    + model.lambda * grad_squ(tile, i, j);

        circulating_current[0][index] = model.zeta * lap * first_y(tile, i, j);
        circulating_current[1][index] = model.zeta * lap * first_x(tile, i, j);

        curandState *rnd = &random_state[index];
        const Scalar mag = sqrt(2 * model.temperature * stencil.dxInv * stencil.dyInv / stencil.dt);
        circulating_current[0][index] += mag * curand_normal(rnd);
        circulating_current[1][index] += mag * curand_normal(rnd);
    }

    __global__ void step(DeviceField field, DeviceField chemical_potential,
                         DeviceCurrent current)
    {
        static constexpr int num_ghost = num_ghost_integrator;

        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load current tile into shared memory.

        __shared__ Scalar mu[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];
        __shared__ Scalar J[d][tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];

        for (int c = 0; c < d; ++c)
        {
            mu[i][j] = chemical_potential[index];
            J[c][i][j] = current[c][index];

            // Fill in ghost points.

            if (threadIdx.y < num_ghost)
            {
                mu[i - num_ghost][j] = chemical_potential[col + ((row - num_ghost + nrows) % nrows) * ncols];
                mu[i + tile_rows][j] = chemical_potential[col + ((row + tile_rows) % nrows) * ncols];

                J[c][i - num_ghost][j] = current[c][col + ((row - num_ghost + nrows) % nrows) * ncols];
                J[c][i + tile_rows][j] = current[c][col + ((row + tile_rows) % nrows) * ncols];
            }

            if (threadIdx.x < num_ghost)
            {
                mu[i][j - num_ghost] = chemical_potential[(col - num_ghost + ncols) % ncols + row * ncols];
                mu[i][j + tile_cols] = chemical_potential[(col + tile_cols) % ncols         + row * ncols];

                J[c][i][j - num_ghost] = current[c][(col - num_ghost + ncols) % ncols + row * ncols];
                J[c][i][j + tile_cols] = current[c][(col + tile_cols) % ncols         + row * ncols];
            }

            if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
            {
                mu[i - num_ghost][j - num_ghost] = chemical_potential[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
                mu[i - num_ghost][j + tile_cols] = chemical_potential[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
                mu[i + tile_rows][j - num_ghost] = chemical_potential[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
                mu[i + tile_rows][j + tile_cols] = chemical_potential[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];

                J[c][i - num_ghost][j - num_ghost] = current[c][(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
                J[c][i - num_ghost][j + tile_cols] = current[c][(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
                J[c][i + tile_rows][j - num_ghost] = current[c][(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
                J[c][i + tile_rows][j + tile_cols] = current[c][(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
            }
        }

        __syncthreads();

        // Integration rule from continuity equation $\partial_t \phi = -\nabla \cdot \vec{J}$:
        Scalar divJ = first_y(J[0], i, j) + first_x(J[1], i, j);
        field[index] -= stencil.dt * (divJ - laplacian(mu, i, j));
    }

    // Basic kernel to check for errors (e.g. if field become nan or inf).
    __global__ void check_finite(DeviceField field, bool* finite)
    {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;
        if (not std::isfinite(field[index])) *finite = false;
    }

    // Seed random number generator on the device.
    __global__ void init_random_state(curandState *state, unsigned long seed)
    {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;
        curand_init(seed, index, 0, &state[index]);
    }
}


/// Host device definitions.

Integrator::Integrator(const HostFieldRef& initial_field,
                       Stencil stencil, Model model)
    : stencil(stencil), model(model),
    nrows(initial_field.rows()),
    ncols(initial_field.cols()),
    pitch_width(initial_field.cols() * sizeof(Scalar)),
    mem_size(initial_field.rows() * initial_field.cols() * sizeof(Scalar))
{
    // Initialise device memory.
    cudaMallocPitch(&field, &field_pitch, pitch_width, nrows);
    cudaMemcpy(field, initial_field.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMallocPitch(&chemical_potential, &chemical_potential_pitch, pitch_width, nrows);
    Field empty = Field::Zero(nrows, ncols);
    for (int c = 0; c < d; ++c)
    {
        cudaMallocPitch(&current[c], &current_pitch[c], pitch_width, nrows);
        cudaMemcpy(current[c], empty.data(), mem_size, cudaMemcpyHostToDevice);
    }

    // Initialise memory for random number generation
    const int n = initial_field.rows() * initial_field.cols();
    cudaMalloc(&random_state, n * sizeof(curandState));

    // Now seed the device for random number generation.

    // Generate a non-deterministic seed. 
    std::random_device rd;
    std::default_random_engine generator{rd()};
    std::uniform_int_distribution<unsigned long long> dist;
    auto seed = dist(generator);
    // Seed the device.
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);
    kernel::init_random_state<<<grid_size, block_dim>>>(random_state, seed);
    cudaDeviceSynchronize();

    kernel::throw_errors();
}

Integrator::Integrator(Integrator&& other) noexcept
    : stencil(other.stencil), model(other.model),
      nrows(other.nrows), ncols(other.ncols),
      pitch_width(other.pitch_width), mem_size(other.mem_size),
      field_pitch(std::move(other.field_pitch)),
      field(std::move(other.field)),
      chemical_potential_pitch(std::move(other.chemical_potential_pitch)),
      chemical_potential(std::move(other.chemical_potential)),
      current_pitch(std::move(other.current_pitch)),
      current(std::move(other.current)),
      random_state(other.random_state),
      timestep(other.timestep),
      timestep_calculated_current(other.timestep_calculated_current)
{
    kernel::throw_errors();
}

Integrator::~Integrator()
{
    cudaFree(field);
    cudaFree(chemical_potential);
    for (int c = 0; c < d; ++c) cudaFree(current[c]);
    cudaFree(random_state);
}

Stencil Integrator::get_stencil() const
{
    return stencil;
}

Model Integrator::get_model() const
{
    return model;
}

HostField Integrator::get_field() const
{
    HostField out(nrows, ncols);
    cudaMemcpy(out.data(), field, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

HostField Integrator::get_chemical_potential()
{
    calculate_current();
    HostField out(nrows, ncols);
    cudaMemcpy(out.data(), chemical_potential, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

HostCurrent Integrator::get_nonconservative_current()
{
    calculate_current();
    HostCurrent out = repeat_array<HostField, d>(nrows, ncols);
    for (int c = 0; c < d; ++c)
        cudaMemcpy(out[c].data(), current[c], mem_size, cudaMemcpyDeviceToHost);
    return out;
}

void Integrator::set_device_parameters()
{
    DeviceStencilParams device_stencil(stencil);
    cudaMemcpyToSymbol(kernel::stencil, &device_stencil, sizeof(DeviceStencilParams));
    cudaMemcpyToSymbol(kernel::nrows, &nrows, sizeof(int));
    cudaMemcpyToSymbol(kernel::ncols, &ncols, sizeof(int));
    cudaMemcpyToSymbol(kernel::model, &model, sizeof(Model));
}

void Integrator::calculate_current()
{
    if (timestep_calculated_current == timestep) return;

    set_device_parameters();
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);
    kernel::calculate_current<<<grid_size, block_dim>>>(field, chemical_potential, current, random_state);
    cudaDeviceSynchronize();
    kernel::throw_errors();

    timestep_calculated_current = timestep;
}

void Integrator::run(int nsteps)
{
    set_device_parameters();
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);


    for (int i = 0; i < nsteps; ++i)
    {
        kernel::calculate_current<<<grid_size, block_dim>>>(field, chemical_potential, current, random_state);
        kernel::step<<<grid_size, block_dim>>>(field, chemical_potential, current);
    }

    cudaDeviceSynchronize();
    kernel::throw_errors();

    // Numerical errors in integration often cause fields to diverge or go to nan, so we
    // need to check for these on the device and raise them up the stack.
    bool finite{true}, *device_finite;
    cudaMalloc(&device_finite, sizeof(bool));
    cudaMemcpy(device_finite, &finite, sizeof(bool), cudaMemcpyHostToDevice);
    kernel::check_finite<<<grid_size, block_dim>>>(field, device_finite);
    cudaMemcpy(&finite, device_finite, sizeof(bool), cudaMemcpyDeviceToHost);

    if (not finite)
    {
        std::string message = "an unknown numerical error occurred during simulation";
        throw kernel::CudaError(message);
    }

    timestep += nsteps;
}
