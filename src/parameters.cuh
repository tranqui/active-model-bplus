#pragma once
#include "parameters.h"

namespace kernel
{
    // Implementation is on a 2d grid with periodic boundary conditions.
    // GPU divided into an (tile_rows x tile_cols) tile (blocks) with
    // a CUDA thread for each tile sharing this memory. Varying the tile size
    // will potentially improve performance on different hardware - I found
    // 16x16 was close to optimum on my machine for simulations on a 1024x1024 grid.
    static constexpr int tile_rows = 16;
    static constexpr int tile_cols = 16;
    // We need ghost points for each tile so we can evaluate derivatives at tile borders.
    static constexpr int num_ghost = 1 + order / 2; // <- minimum for fourth derivatives

    // Stencil parameters - 2d space (x, y), and time t.
    extern __constant__ DeviceStencilParams stencil;
    extern __constant__ int nrows, ncols; // number of points in spatial grid
    extern __constant__ Model model;
}
