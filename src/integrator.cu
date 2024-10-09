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

    __global__ void calculate_forces(DeviceField field,
                                     DeviceField passive_chemical_potential,
                                     DeviceField active_chemical_potential,
                                     DeviceCurrent circulating_current,
                                     DeviceCurrent random_current,
                                     curandState *random_state)
    {
        static constexpr int num_ghost = 1 + order/2;

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

        Scalar lap = tjhung_laplacian(tile, i, j);
        Scalar grad_y = tjhung_first_y(tile, i, j);
        Scalar grad_x = tjhung_first_x(tile, i, j);
        passive_chemical_potential[index] =
            bulk_chemical_potential(tile[i][j]) - model.kappa * lap;
        active_chemical_potential[index] = model.lambda * (square(grad_y) + square(grad_x));

        circulating_current[0][index] = model.zeta * lap * grad_y;
        circulating_current[1][index] = model.zeta * lap * grad_x;

        curandState *rnd = &random_state[index];
        Scalar mag = model.noise_strength * stencil.noise_strength;
        random_current[0][index] = mag * curand_normal(rnd);
        random_current[1][index] = mag * curand_normal(rnd);
    }

    __global__ void calculate_local_currents(DeviceField passive_chemical_potential,
                                             DeviceField active_chemical_potential,
                                             DeviceCurrent passive_current,
                                             DeviceCurrent active_current)
    {
        static constexpr int num_ghost = order/2;

        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load chemical potential tiles into shared memory.

        __shared__ Scalar mup[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];
        __shared__ Scalar mua[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];
        mup[i][j] = passive_chemical_potential[index];
        mua[i][j] = active_chemical_potential[index];

        // Fill in ghost points.

        if (threadIdx.y < num_ghost)
        {
            mup[i - num_ghost][j] = passive_chemical_potential[col + ((row - num_ghost + nrows) % nrows) * ncols];
            mup[i + tile_rows][j] = passive_chemical_potential[col + ((row + tile_rows) % nrows) * ncols];

            mua[i - num_ghost][j] = active_chemical_potential[col + ((row - num_ghost + nrows) % nrows) * ncols];
            mua[i + tile_rows][j] = active_chemical_potential[col + ((row + tile_rows) % nrows) * ncols];
        }

        if (threadIdx.x < num_ghost)
        {
            mup[i][j - num_ghost] = passive_chemical_potential[(col - num_ghost + ncols) % ncols + row * ncols];
            mup[i][j + tile_cols] = passive_chemical_potential[(col + tile_cols) % ncols         + row * ncols];

            mua[i][j - num_ghost] = active_chemical_potential[(col - num_ghost + ncols) % ncols + row * ncols];
            mua[i][j + tile_cols] = active_chemical_potential[(col + tile_cols) % ncols         + row * ncols];
        }

        if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
        {
            mup[i - num_ghost][j - num_ghost] = passive_chemical_potential[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
            mup[i - num_ghost][j + tile_cols] = passive_chemical_potential[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
            mup[i + tile_rows][j - num_ghost] = passive_chemical_potential[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
            mup[i + tile_rows][j + tile_cols] = passive_chemical_potential[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];

            mua[i - num_ghost][j - num_ghost] = active_chemical_potential[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
            mua[i - num_ghost][j + tile_cols] = active_chemical_potential[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
            mua[i + tile_rows][j - num_ghost] = active_chemical_potential[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
            mua[i + tile_rows][j + tile_cols] = active_chemical_potential[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
        }

        __syncthreads();

        passive_current[0][index] = -first_y(mup, i, j);
        passive_current[1][index] = -first_x(mup, i, j);

        active_current[0][index] = -tjhung_first_y(mua, i, j);
        active_current[1][index] = -tjhung_first_x(mua, i, j);
    }

    __global__ void step(DeviceField field,
                         DeviceCurrent pass_current,
                         DeviceCurrent lamb_current,
                         DeviceCurrent circ_current,
                         DeviceCurrent rand_current)
    {
        static constexpr int num_ghost = order/2;

        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load current tiles into shared memory.

        __shared__ Scalar J1[d][tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];
        __shared__ Scalar J2[d][tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];

        for (int c = 0; c < d; ++c)
        {
            J1[c][i][j] = pass_current[c][index] + rand_current[c][index];
            J2[c][i][j] = lamb_current[c][index] + circ_current[c][index];

            if (threadIdx.y < num_ghost)
            {
                J1[c][i - num_ghost][j] = pass_current[c][col + ((row - num_ghost + nrows) % nrows) * ncols]
                                        + rand_current[c][col + ((row - num_ghost + nrows) % nrows) * ncols];
                J1[c][i + tile_rows][j] = pass_current[c][col + ((row + tile_rows) % nrows) * ncols]
                                        + rand_current[c][col + ((row + tile_rows) % nrows) * ncols];

                J2[c][i - num_ghost][j] = lamb_current[c][col + ((row - num_ghost + nrows) % nrows) * ncols]
                                        + circ_current[c][col + ((row - num_ghost + nrows) % nrows) * ncols];
                J2[c][i + tile_rows][j] = lamb_current[c][col + ((row + tile_rows) % nrows) * ncols]
                                        + circ_current[c][col + ((row + tile_rows) % nrows) * ncols];

            }

            if (threadIdx.x < num_ghost)
            {
                J1[c][i][j - num_ghost] = pass_current[c][(col - num_ghost + ncols) % ncols + row * ncols]
                                        + rand_current[c][(col - num_ghost + ncols) % ncols + row * ncols];
                J1[c][i][j + tile_cols] = pass_current[c][(col + tile_cols) % ncols         + row * ncols]
                                        + rand_current[c][(col + tile_cols) % ncols         + row * ncols];

                J2[c][i][j - num_ghost] = lamb_current[c][(col - num_ghost + ncols) % ncols + row * ncols]
                                        + circ_current[c][(col - num_ghost + ncols) % ncols + row * ncols];
                J2[c][i][j + tile_cols] = lamb_current[c][(col + tile_cols) % ncols         + row * ncols]
                                        + circ_current[c][(col + tile_cols) % ncols         + row * ncols];

            }

            if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
            {
                J1[c][i - num_ghost][j - num_ghost] = pass_current[c][(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols]
                                                    + rand_current[c][(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
                J1[c][i - num_ghost][j + tile_cols] = pass_current[c][(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols]
                                                    + rand_current[c][(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
                J1[c][i + tile_rows][j - num_ghost] = pass_current[c][(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols]
                                                    + rand_current[c][(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
                J1[c][i + tile_rows][j + tile_cols] = pass_current[c][(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols]
                                                    + rand_current[c][(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];

                J2[c][i - num_ghost][j - num_ghost] = lamb_current[c][(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols]
                                                    + circ_current[c][(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
                J2[c][i - num_ghost][j + tile_cols] = lamb_current[c][(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols]
                                                    + circ_current[c][(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
                J2[c][i + tile_rows][j - num_ghost] = lamb_current[c][(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols]
                                                    + circ_current[c][(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
                J2[c][i + tile_rows][j + tile_cols] = lamb_current[c][(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols]
                                                    + circ_current[c][(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
            }
        }

        __syncthreads();

        // Integration rule from continuity equation $\partial_t \phi = -\nabla \cdot \vec{J}$:
        Scalar divJ = tjhung_first_y(J2[0], i, j) + tjhung_first_x(J2[1], i, j)
                    + first_y(J1[0], i, j) + first_x(J1[1], i, j);
        field[index] -= stencil.dt * divJ;
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
    cudaMallocPitch(&passive_chemical_potential, &passive_chemical_potential_pitch, pitch_width, nrows);
    cudaMallocPitch(&active_chemical_potential, &active_chemical_potential_pitch, pitch_width, nrows);
    Field empty = Field::Zero(nrows, ncols);
    for (int c = 0; c < d; ++c)
    {
        cudaMallocPitch(&pass_current[c], &pass_current_pitch[c], pitch_width, nrows);
        cudaMallocPitch(&lamb_current[c], &lamb_current_pitch[c], pitch_width, nrows);
        cudaMallocPitch(&circ_current[c], &circ_current_pitch[c], pitch_width, nrows);
        cudaMallocPitch(&rand_current[c], &rand_current_pitch[c], pitch_width, nrows);
        cudaMemcpy(pass_current[c], empty.data(), mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(lamb_current[c], empty.data(), mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(circ_current[c], empty.data(), mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(rand_current[c], empty.data(), mem_size, cudaMemcpyHostToDevice);
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
      passive_chemical_potential_pitch(std::move(other.passive_chemical_potential_pitch)),
      active_chemical_potential_pitch(std::move(other.active_chemical_potential_pitch)),
      passive_chemical_potential(std::move(other.passive_chemical_potential)),
      active_chemical_potential(std::move(other.active_chemical_potential)),
      pass_current_pitch(std::move(other.pass_current_pitch)),
      lamb_current_pitch(std::move(other.lamb_current_pitch)),
      circ_current_pitch(std::move(other.circ_current_pitch)),
      rand_current_pitch(std::move(other.rand_current_pitch)),
      pass_current(std::move(other.pass_current)),
      lamb_current(std::move(other.lamb_current)),
      circ_current(std::move(other.circ_current)),
      rand_current(std::move(other.rand_current)),
      random_state(other.random_state),
      timestep(other.timestep),
      timestep_calculated_current(other.timestep_calculated_current)
{
    kernel::throw_errors();
}

Integrator::~Integrator()
{
    cudaFree(field);
    cudaFree(passive_chemical_potential);
    cudaFree(active_chemical_potential);
    for (int c = 0; c < d; ++c)
    {
        cudaFree(pass_current[c]);
        cudaFree(lamb_current[c]);
        cudaFree(circ_current[c]);
        cudaFree(rand_current[c]);
    }
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
    return get_passive_chemical_potential() + get_active_chemical_potential();
}

HostField Integrator::get_passive_chemical_potential()
{
    calculate_current();
    HostField out(nrows, ncols);
    cudaMemcpy(out.data(), passive_chemical_potential, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

HostField Integrator::get_active_chemical_potential()
{
    calculate_current();
    HostField out(nrows, ncols);
    cudaMemcpy(out.data(), active_chemical_potential, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

HostCurrent Integrator::get_current()
{
    HostCurrent random = get_random_current();
    HostCurrent total = get_deterministic_current();
    for (int c = 0; c < d; ++c) total[c] += random[c];
    return total;
}


HostCurrent Integrator::get_deterministic_current()
{
    HostCurrent passive = get_passive_current();
    HostCurrent total = get_active_current();
    for (int c = 0; c < d; ++c) total[c] += passive[c];
    return total;
}

HostCurrent Integrator::get_passive_current()
{
    calculate_current();
    HostCurrent out = repeat_array<HostField, d>(nrows, ncols);
    for (int c = 0; c < d; ++c)
        cudaMemcpy(out[c].data(), pass_current[c], mem_size, cudaMemcpyDeviceToHost);
    return out;
}

HostCurrent Integrator::get_active_current()
{
    HostCurrent local = get_lambda_current();
    HostCurrent total = get_circulating_current();
    for (int c = 0; c < d; ++c) total[c] += local[c];
    return total;
}

HostCurrent Integrator::get_lambda_current()
{
    calculate_current();
    HostCurrent out = repeat_array<HostField, d>(nrows, ncols);
    for (int c = 0; c < d; ++c)
        cudaMemcpy(out[c].data(), lamb_current[c], mem_size, cudaMemcpyDeviceToHost);
    return out;
}

HostCurrent Integrator::get_circulating_current()
{
    calculate_current();
    HostCurrent out = repeat_array<HostField, d>(nrows, ncols);
    for (int c = 0; c < d; ++c)
        cudaMemcpy(out[c].data(), circ_current[c], mem_size, cudaMemcpyDeviceToHost);
    return out;
}

HostCurrent Integrator::get_random_current()
{
    calculate_current();
    HostCurrent out = repeat_array<HostField, d>(nrows, ncols);
    for (int c = 0; c < d; ++c)
        cudaMemcpy(out[c].data(), rand_current[c], mem_size, cudaMemcpyDeviceToHost);
    return out;
}

void Integrator::set_device_parameters()
{
    DeviceStencilParams device_stencil(stencil);
    cudaMemcpyToSymbol(kernel::stencil, &device_stencil, sizeof(DeviceStencilParams));
    cudaMemcpyToSymbol(kernel::nrows, &nrows, sizeof(int));
    cudaMemcpyToSymbol(kernel::ncols, &ncols, sizeof(int));
    DeviceModelParams device_model(model);
    cudaMemcpyToSymbol(kernel::model, &device_model, sizeof(DeviceModelParams));
}

void Integrator::calculate_current()
{
    if (timestep_calculated_current == timestep) return;

    set_device_parameters();
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);
    kernel::calculate_forces<<<grid_size, block_dim>>>(
        field, passive_chemical_potential, active_chemical_potential,
        circ_current, rand_current, random_state);
    kernel::calculate_local_currents<<<grid_size, block_dim>>>(
        passive_chemical_potential, active_chemical_potential,
        pass_current, lamb_current);
    cudaDeviceSynchronize();
    kernel::throw_errors();

    timestep_calculated_current = timestep;
}

void Integrator::run(int nsteps)
{
    // Ensure forces/currents are pre-calculated at current timestep.
    calculate_current();

    set_device_parameters();
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);

    for (int i = 0; i < nsteps; ++i)
    {
        kernel::step<<<grid_size, block_dim>>>(
            field, pass_current, lamb_current, circ_current, rand_current);
        // Calculate for next step, or to be read by user if stopping.
        kernel::calculate_forces<<<grid_size, block_dim>>>(
            field, passive_chemical_potential, active_chemical_potential,
            circ_current, rand_current, random_state);
        kernel::calculate_local_currents<<<grid_size, block_dim>>>(
            passive_chemical_potential, active_chemical_potential,
            pass_current, lamb_current);
    }

    timestep += nsteps;

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
}
