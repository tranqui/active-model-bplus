#pragma once
#include <Eigen/Eigen>

static constexpr int d = 2;      // number of spatial dimensions
using Scalar = double;           // type for numeric data

using Field = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FieldRef = Eigen::Ref<const Field>;

using HostField = Field;
using HostFieldRef = FieldRef;
using DeviceField = Scalar*;

using Gradient = std::array<Field, d>;
using HostGradient = Gradient;
using DeviceGradient = std::array<Scalar*, d>;

using Current = Gradient;
using HostCurrent = HostGradient;
using DeviceCurrent = DeviceGradient;

// Direction of index offset when derivatives are taken on staggered grids.
// See `StaggeredDerivative` in finite_differences.cuh for more info.
enum StaggeredGridDirection { Left, Central, Right };
