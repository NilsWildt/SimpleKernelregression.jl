module Kernelregression
# Defines different kernels that can be used for regularized kernel interpolation (kernel regression)
# Using: Approximation with Kernel Methods WS 17/18 by Dr. Gabriele Santin & Prof. Haasdonk, Univ. of Stuttgart
using LinearAlgebra
using ConcreteStructs: @concrete
using TestItems
abstract type SKernel end
export get_kernel_interpolant, get_kernel_derivative_interpolant, SKernel, Gaussian, Imq, Linear, Polynomial, Mq, Epanechnikov


include("k_gaussian.jl")
include("k_polynomial.jl")
include("k_linear.jl")
include("k_imq.jl")
include("k_mq.jl")
include("k_epanechnikov.jl")


function prepend_one(X::AbstractArray)
    return X = cat(ones(size(X, 1)), X; dims = 2)
end

"""
pDist2
pairwise euclidean distance
```math
(x - y) = sqrt(x ^ 2 + y ^ 2 - 2 * x * y)
```
"""
function pDist2(X::AbstractArray, Y::AbstractArray)
    Dsq = pDist2Squared(X, Y)
    return sqrt.(max.(Dsq, zero(eltype(Dsq))))
end


function pDist2Squared(X::AbstractArray{T}, Y::AbstractArray{R}) where {T <: Real, R <: Real}
    # (x - y) ^ 2 = x ^ 2 + y ^ 2 - 2 * x * y
    Ly = size(Y, 1)
    Lx = size(X, 1)
    S = promote_type(T, R)

    # Precompute row-wise sums of squares
    x_sq = sum(abs2, X; dims = 2)
    y_sq = sum(abs2, Y; dims = 2)

    # Expand to matrices without allocating large ones of ones()
    T1 = Matrix{S}(undef, Lx, Ly)
    T2 = Matrix{S}(undef, Lx, Ly)
    @inbounds for j in 1:Ly
        for i in 1:Lx
            T1[i, j] = x_sq[i]
            T2[i, j] = y_sq[j]
        end
    end

    T3 = -2 .* (X * Y')
    return T1 + T2 + T3
end


# Inner-product matrix helper for linear/polynomial kernels
function kernel_dot(X::AbstractArray{T}, Y::AbstractArray{R}) where {T <: Real, R <: Real}
    return X * Y'
end

# If Y is already an Adjoint, avoid double-transpose
function kernel_dot(X::AbstractArray{T}, Y::Adjoint) where {T <: Real}
    return X * Y
end


function regularize!(K::AbstractMatrix, reg::Real = 1.0e-6)
    return K .+= reg .* Matrix(I, size(K, 1), size(K, 2))
end

function regularize(K::AbstractMatrix, reg::Real = 1.0e-6)
    return K .+ reg .* Matrix(I, size(K, 1), size(K, 2))
end


function get_kernel_interpolant(X::AbstractArray, Y::AbstractArray, mykernel::SKernel, reg::Real = 1.0e-6)
    K = evalKmatrix(mykernel, X, X)
    K = regularize(K, reg)
    facts = (K \ Y)
    fxnew(xnew) = evalKmatrix(mykernel, xnew, X) * facts
    return fxnew
end

function get_kernel_derivative_interpolant(X::AbstractArray, Y::AbstractArray, mykernel::SKernel, reg::Real = 1.0e-6)
    K = evalKmatrix(mykernel, X, X)
    K = regularize(K, reg)
    facts = (K \ Y)
    fxnew_derivative(xnew) = evalKmatrix_derivative(mykernel, xnew, X) * facts
    return fxnew_derivative
end


# Optional: pre-factorized interpolant types for fast updates of Y
@concrete mutable struct KernelInterpolant{T, KT <: SKernel, FT}
    X::AbstractArray{T}
    kernel::KT
    reg::T
    K_factorization::FT
    coefficients::AbstractArray{T}
    initialized::Bool
end

function KernelInterpolant(X::AbstractArray{T}, kernel::SKernel, reg::Real = 1.0e-6) where {T <: Real}
    K = evalKmatrix(kernel, X, X)
    K = regularize(K, reg)
    Kf = factorize(K)
    coeffs = Array{T}(undef, 0, 0)
    return KernelInterpolant{T, typeof(kernel), typeof(Kf)}(X, kernel, T(reg), Kf, coeffs, true)
end

@inline function solve_matrix(Kf, Y::AbstractArray{T}) where {T <: Real}
    return Kf \ Y
end

function update_coefficients!(interp::KernelInterpolant{R}, Y::AbstractArray{T}) where {R, T <: Real}
    interp.coefficients = solve_matrix(interp.K_factorization, Y)
    return interp
end

function (interp::KernelInterpolant{T})(xnew::AbstractArray) where {T <: Real}
    return evalKmatrix(interp.kernel, xnew, interp.X) * interp.coefficients
end


@concrete mutable struct KernelDerivativeInterpolant{T, KT <: SKernel, FT}
    X::AbstractArray{T}
    kernel::KT
    reg::T
    K_factorization::FT
    coefficients::AbstractArray{T}
    initialized::Bool
end

function KernelDerivativeInterpolant(X::AbstractArray{T}, kernel::SKernel, reg::Real = 1.0e-6) where {T <: Real}
    K = evalKmatrix(kernel, X, X)
    K = regularize(K, reg)
    Kf = factorize(K)
    coeffs = Array{T}(undef, 0, 0)
    return KernelDerivativeInterpolant{T, typeof(kernel), typeof(Kf)}(X, kernel, T(reg), Kf, coeffs, true)
end

function update_coefficients!(interp::KernelDerivativeInterpolant{T}, Y::AbstractArray{T}) where {T <: Real}
    interp.coefficients = interp.K_factorization \ Y
    return interp
end

function (interp::KernelDerivativeInterpolant{T})(xnew::AbstractArray) where {T <: Real}
    return evalKmatrix_derivative(interp.kernel, xnew, interp.X) * interp.coefficients
end


# Derivatives for selected kernels
function evalKmatrix_derivative(gaussian::Gaussian, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    K = _evalKmatrix(gaussian, xy_dist)
    dK = @. (-1.0 / gaussian.σ^2) * xy_dist * K
    return dK
end

function evalKmatrix_derivative(imq::Imq, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    # ϕ(r) = (1 + (r/σ)^2)^(-1/2); derivative wrt r^2 scales similarly for matrix form
    dϕ(r) = -0.5 * (1 + (r / imq.σ)^2)^(-1.5) * (1 / imq.σ^2)
    return dϕ.(xy_dist)
end


function marginal_log_likelihood(θ, X::AbstractArray{T}, Y::AbstractArray{R}, KernelType, reg::Real = 1.0e-5) where {T <: Real, R <: Real}
    σ = θ[1]
    kernel = KernelType(σ)
    K = evalKmatrix(kernel, X, X) .+ reg .* I
    L = nothing
    try
        L = cholesky(K).L
    catch
        @warn "Was not able to compute Cholesky decomposition"
        return Inf
    end
    log_likelihood = 0.0
    for i in 1:size(Y, 2)
        α = L' \ (L \ Y[:, i])
        log_likelihood -= 0.5 * (Y[:, i]' * α) + sum(log.(diag(L))) + 0.5 * size(Y, 1) * log(2 * π)
    end
    return -log_likelihood
end

# Inline tests using TestItems
@testitem "Gaussian Kernel Construction" begin
    σ = 1.0
    kernel = Gaussian(σ)
    @test kernel.σ == σ
    @test typeof(kernel) <: SKernel
end

@testitem "IMQ Kernel Construction" begin
    σ = 1.0
    kernel = Imq(σ)
    @test kernel.σ == σ
    @test typeof(kernel) <: SKernel
end

@testitem "Kernel Matrix Evaluation" begin
    using Random
    Random.seed!(42)
    X = rand(10, 2)
    Y = rand(5, 2)
    σ = 1.0
    gaussian = Gaussian(σ)
    K_gaussian = SimpleKernelRegression.evalKmatrix(gaussian, X, Y)
    @test size(K_gaussian) == (10, 5)
    @test all(isfinite, K_gaussian)
    imq = Imq(σ)
    K_imq = SimpleKernelRegression.evalKmatrix(imq, X, Y)
    @test size(K_imq) == (10, 5)
    @test all(isfinite, K_imq)
end

@testitem "Kernel Derivative Evaluation" begin
    using Random
    Random.seed!(42)
    X = rand(10, 2)
    Y = rand(5, 2)
    σ = 1.0
    gaussian = Gaussian(σ)
    dK_gaussian = SimpleKernelRegression.evalKmatrix_derivative(gaussian, X, Y)
    @test size(dK_gaussian) == (10, 5)
    @test all(isfinite, dK_gaussian)
    imq = Imq(σ)
    dK_imq = SimpleKernelRegression.evalKmatrix_derivative(imq, X, Y)
    @test size(dK_imq) == (10, 5)
    @test all(isfinite, dK_imq)
end

@testitem "Kernel Interpolant" begin
    using Random
    Random.seed!(42)
    X = rand(10, 2)
    Y = rand(10, 1)
    σ = 1.0
    reg = 1.0e-6
    gaussian = Gaussian(σ)
    interpolant = SimpleKernelRegression.get_kernel_interpolant(X, Y, gaussian, reg)
    X_test = rand(5, 2)
    Y_pred = interpolant(X_test)
    @test size(Y_pred) == (5, 1)
    @test all(isfinite, Y_pred)
    imq = Imq(σ)
    interpolant2 = SimpleKernelRegression.get_kernel_interpolant(X, Y, imq, reg)
    Y_pred2 = interpolant2(X_test)
    @test size(Y_pred2) == (5, 1)
    @test all(isfinite, Y_pred2)
end

@testitem "Kernel Derivative Interpolant" begin
    using Random
    Random.seed!(42)
    X = rand(10, 2)
    Y = rand(10, 1)
    σ = 1.0
    reg = 1.0e-6
    gaussian = Gaussian(σ)
    interpolant = SimpleKernelRegression.get_kernel_derivative_interpolant(X, Y, gaussian, reg)
    X_test = rand(5, 2)
    Y_pred = interpolant(X_test)
    @test size(Y_pred) == (5, 1)
    @test all(isfinite, Y_pred)
    imq = Imq(σ)
    interpolant2 = SimpleKernelRegression.get_kernel_derivative_interpolant(X, Y, imq, reg)
    Y_pred2 = interpolant2(X_test)
    @test size(Y_pred2) == (5, 1)
    @test all(isfinite, Y_pred2)
end

@testitem "Pairwise Distance Functions" begin
    using Random
    Random.seed!(42)
    X = rand(10, 10)
    Y = rand(10, 10)
    D_squared = SimpleKernelRegression.pDist2Squared(X, Y)
    @test size(D_squared) == (10, 10)
    @test all(isfinite, D_squared)
    @test all(≥(0), D_squared)
    D = SimpleKernelRegression.pDist2(X, Y)
    @test size(D) == (10, 10)
    @test all(isfinite, D)
end

@testitem "Marginal Log Likelihood" begin
    using Random
    Random.seed!(42)
    X = rand(20, 10)
    Y = rand(20, 20)
    θ = [1.0]
    ll_gaussian = SimpleKernelRegression.marginal_log_likelihood(θ, X, Y, Gaussian)
    @test isfinite(ll_gaussian)
    ll_imq = SimpleKernelRegression.marginal_log_likelihood(θ, X, Y, Imq)
    @test isfinite(ll_imq)
end

end # End module
