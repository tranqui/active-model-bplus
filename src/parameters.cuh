#pragma once
#include "parameters.h"

namespace kernel
{
    // Stencil parameters - 2d space (x, y), and time t.
    extern __constant__ DeviceStencilParams stencil;
    extern __constant__ int nrows, ncols; // number of points in spatial grid
    extern __constant__ DeviceModelParams model;
}
