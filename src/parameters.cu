#include "parameters.cuh"

namespace kernel
{
    // Stencil parameters - 2d space (x, y), and time t.
    __constant__ DeviceStencilParams stencil;
    __constant__ int nrows, ncols;        // number of points in spatial grid
    __constant__ Model model;
}
