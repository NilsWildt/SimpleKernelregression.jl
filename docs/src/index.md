```@meta
CurrentModule = SimpleKernelRegression
```

# SimpleKernelRegression.jl

A Julia package for kernel regression (regularized kernel interpolation) using various kernel functions.

## Overview

SimpleKernelRegression provides a collection of kernel functions and utilities for performing kernel-based regression. The package supports multiple kernel types and provides both simple functional interfaces and optimized struct-based implementations for performance-critical applications.

## Installation

```julia
using Pkg
Pkg.add("SimpleKernelRegression")
```

## Quick Start

```@example
using SimpleKernelRegression
using Random

# Generate sample data
Random.seed!(42)
X_train = rand(20, 2)  # 20 training points in 2D
Y_train = sum(X_train.^2, dims=2) + 0.1 * randn(20, 1)  # Quadratic function with noise

# Create a Gaussian kernel
kernel = Gaussian(1.0)

# Get interpolant function
f = get_kernel_interpolant(X_train, Y_train, kernel; reg=1e-6)

# Make predictions
X_test = rand(10, 2)
Y_pred = f(X_test)
```

## Available Kernels

The package provides several kernel types:

- **Gaussian Kernel**: `Gaussian(σ)` - RBF kernel with bandwidth σ
- **Linear Kernel**: `Linear(calc_bias)` - Inner product kernel with optional bias
- **Polynomial Kernel**: `Polynomial(a, p)` - Polynomial kernel with offset a and degree p
- **Inverse Multiquadratic**: `Imq(σ)` - IMQ kernel with scaling σ
- **Multiquadratic**: `Mq(σ)` - MQ kernel with scaling σ
- **Epanechnikov**: `Epanechnikov(σ)` - Local kernel with compact support

## Solver Options

The package provides three solver methods for the kernel system:

- `BackslashSolver` (default): Standard LU factorization
- `PinvSolver`: Pseudoinverse (robust for ill-conditioned systems)
- `LevenbergMarquardtSolver`: Adaptive regularization with Levenberg-Marquardt

Example with different solver:
```julia
f = get_kernel_interpolant(X_train, Y_train, kernel; solver=PinvSolver)
```

## Features

- **Multiple Kernel Types**: Wide variety of kernel functions
- **Flexible Interface**: Simple functional API and advanced struct-based API
- **Regularization**: Built-in regularization to handle ill-conditioned problems
- **Derivatives**: Support for kernel derivative computation
- **Optimized Performance**: Uses platform-specific BLAS libraries
- **Robust Solvers**: Multiple numerical methods for different conditioning scenarios

## Mathematical Background

The package implements regularized kernel interpolation:

```math
f(x) = \sum_{i=1}^n \alpha_i K(x, x_i)
```

where the coefficients α are found by solving:
```math
(K + \lambda I)\alpha = y
```

with K being the kernel matrix, λ the regularization parameter, and y the target values.

