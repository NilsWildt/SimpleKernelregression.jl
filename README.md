# SimpleKernelRegression

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://NilsWildt.github.io/SimpleKernelRegression.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://NilsWildt.github.io/SimpleKernelRegression.jl/dev)
[![Test workflow status](https://github.com/NilsWildt/SimpleKernelRegression.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/NilsWildt/SimpleKernelRegression.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![codecov](https://codecov.io/github/NilsWildt/SimpleKernelregression.jl/graph/badge.svg?token=AK3U2LS9BZ)](https://codecov.io/github/NilsWildt/SimpleKernelregression.jl)
[![Docs workflow Status](https://github.com/NilsWildt/SimpleKernelRegression.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/NilsWildt/SimpleKernelRegression.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)


![Kernel Regression Example](kernel_regression_3d_example.png)

## Overview

SimpleKernelRegression is just my own simple kernel regression. Doesn't do more, doesn't do less. Not everything is yet throughfully tested -- Interpolation and Regression are solid.

## Main sources

* [https://gabrielesantin.github.io/files/approximation_with_kernel_methods.pdf](https://gabrielesantin.github.io/files/approximation_with_kernel_methods.pdf)
* [https://github.com/GabrieleSantin/VKOGA/tree/master)](https://github.com/GabrieleSantin/VKOGA/tree/master)

Further Literature
1. H. Wendland. "Scattered Data Approximation", Cambridge Monographs on Applied and Computational Mathematics, 2005.
2. B. Schölkopf and A. J. Smola. "Learning with Kernels: Support Vector Machines, Regularization, Optimization and Beyond", MIT Press, 2002.
3. G. E. Fasshauer. "Meshfree Approximation Methods with MATLAB", World Scientific Publishing, 2007.


## Installation

```julia
using Pkg
Pkg.add("SimpleKernelRegression")
```

## Quick Start

```julia
using SimpleKernelRegression
using Random

# Generate sample data
Random.seed!(42)
X_train = rand(20, 2)  # 20 training points in 2D
Y_train = sum(X_train.^2, dims=2) + 0.1 * randn(20, 1)  # Quadratic function with noise

# Create a Gaussian kernel
kernel = Gaussian(1.0)

# Get interpolant function
f = get_kernel_interpolant(X_train, Y_train, kernel, 1e-6)

# Make predictions
X_test = rand(10, 2)
Y_pred = f(X_test)
```

## Available Kernels

### Gaussian Kernel
```julia
kernel = Gaussian(σ)  # σ is the bandwidth parameter
```
Implements: `K(x,y) = exp(-||x-y||²/(2σ²))`

### Linear Kernel
```julia
kernel = Linear(calc_bias)  # calc_bias determines if bias term is included
```
Implements: `K(x,y) = ⟨x,y⟩` (or `⟨[1;x],[1;y]⟩` with bias)

### Polynomial Kernel
```julia
kernel = Polynomial(a, p)  # a is additive constant, p is degree
```
Implements: `K(x,y) = (⟨x,y⟩ + a)^p`

### Inverse Multiquadratic (IMQ) Kernel
```julia
kernel = Imq(σ)  # σ is the scaling parameter
```
Implements: `K(x,y) = (1 + ||x-y||²/σ²)^(-1/2)`

### Multiquadratic (MQ) Kernel
```julia
kernel = Mq(σ)  # σ is the scaling parameter
```
Implements: `K(x,y) = (1 + σ²||x-y||²)^(1/2)`

### Epanechnikov Kernel
```julia
kernel = Epanechnikov(σ)  # σ is the scaling parameter
```
Implements a local kernel with compact support.

## Advanced Usage

### Optimized Interpolants
For repeated predictions with the same kernel and training data but different target values:

```julia
# Create a reusable interpolant structure
interp = KernelInterpolant(X_train, kernel, 1e-6)

# Update with new target values
update_coefficients!(interp, Y_train)

# Make predictions
Y_pred = interp(X_test)
```

### Kernel Derivatives
For kernels that support derivative computation:

```julia
# Get derivative interpolant
f_deriv = get_kernel_derivative_interpolant(X_train, Y_train, gaussian_kernel, 1e-6)

# Compute derivatives at test points
dY_pred = f_deriv(X_test)
```


## Contributing

Contributions are welcome! Please see the documentation for development guidelines.

## License

This project is licensed under the MIT License.
