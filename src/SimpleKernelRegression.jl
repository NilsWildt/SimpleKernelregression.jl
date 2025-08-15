module SimpleKernelRegression
# Defines different kernels that can be used for regularized kernel interpolation (kernel regression)
# Using: Approximation with Kernel Methods WS 17/18 by Dr. Gabriele Santin & Prof. Haasdonk, Univ. of Stuttgart
using LinearAlgebra
using ConcreteStructs: @concrete
const CPU_MODEL = get(
    ENV, "CPU_MODEL", try
        Sys.cpu_info()[1].model
    catch
        ""
    end
)
if Sys.isapple() && Sys.ARCH in (:aarch64, :arm64)
    @info "Using `AppleAccelerate.jl` for Apple Silicon."
    using AppleAccelerate
elseif Sys.ARCH == :x86_64 && occursin(r"intel"i, CPU_MODEL)
    @info "Detected Intel x86_64 CPU. Loading `MKL.jl`."
    using MKL
end


using TestItems

abstract type SKernel end

"""
Solver methods for kernel regression systems.
"""
@enum SolverMethod begin
    BackslashSolver  # Default: K \\ Y (LU factorization)
    PinvSolver      # Pseudoinverse: pinv(K) * Y
    LevenbergMarquardtSolver  # Levenberg-Marquardt with adaptive regularization
end

export get_kernel_interpolant, get_kernel_derivative_interpolant, SKernel, Gaussian, Imq, Linear, Polynomial, Mq, Epanechnikov
export SolverMethod, BackslashSolver, PinvSolver, LevenbergMarquardtSolver


include("k_gaussian.jl")
include("k_polynomial.jl")
include("k_linear.jl")
include("k_imq.jl")
include("k_mq.jl")
include("k_epanechnikov.jl")


"""
    prepend_one(X::AbstractArray)

Prepend a column of ones to the input matrix X for bias terms in linear kernels.

# Arguments
- `X::AbstractArray`: Input matrix (n × d)

# Returns
- Matrix of size (n, d+1) with ones prepended as the first column
"""
function prepend_one(X::AbstractArray)
    return X = cat(ones(size(X, 1)), X; dims = 2)
end

"""
    pDist2(X::AbstractArray, Y::AbstractArray)

Compute pairwise Euclidean distances between points in X and Y.

Uses the identity: ||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩

# Arguments
- `X::AbstractArray`: First set of points (n × d matrix)
- `Y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing pairwise distances ||xᵢ - yⱼ||
"""
function pDist2(X::AbstractArray, Y::AbstractArray)
    Dsq = pDist2Squared(X, Y)
    return sqrt.(max.(Dsq, zero(eltype(Dsq))))
end


"""
    pDist2Squared(X::AbstractArray{T}, Y::AbstractArray{R}) where {T <: Real, R <: Real}

Compute pairwise squared Euclidean distances between points in X and Y.

Efficiently computes ||x - y||² using the identity:
||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩

# Arguments
- `X::AbstractArray{T}`: First set of points (n × d matrix)
- `Y::AbstractArray{R}`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing pairwise squared distances ||xᵢ - yⱼ||²
"""
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


"""
    kernel_dot(X::AbstractArray{T}, Y::AbstractArray{R}) where {T <: Real, R <: Real}

Compute inner product matrix for linear/polynomial kernels.

# Arguments
- `X::AbstractArray{T}`: First set of points (n × d matrix)
- `Y::AbstractArray{R}`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing inner products ⟨xᵢ, yⱼ⟩
"""
function kernel_dot(X::AbstractArray{T}, Y::AbstractArray{R}) where {T <: Real, R <: Real}
    return X * Y'
end

"""
    kernel_dot(X::AbstractArray{T}, Y::Adjoint) where {T <: Real}

Optimized version when Y is already transposed to avoid double-transpose.
"""
function kernel_dot(X::AbstractArray{T}, Y::Adjoint) where {T <: Real}
    return X * Y
end


"""
    regularize!(K::AbstractMatrix, reg::Real = 1.0e-6)

Add regularization to kernel matrix K in-place: K := K + reg*I.

# Arguments
- `K::AbstractMatrix`: Kernel matrix to regularize (modified in-place)
- `reg::Real`: Regularization parameter (default: 1e-6)

# Returns
- The modified matrix K
"""
function regularize!(K::AbstractMatrix, reg::Real = 1.0e-6)
    return K .+= reg .* Matrix(I, size(K, 1), size(K, 2))
end

"""
    regularize(K::AbstractMatrix, reg::Real = 1.0e-6)

Add regularization to kernel matrix K: returns K + reg*I.

# Arguments
- `K::AbstractMatrix`: Kernel matrix
- `reg::Real`: Regularization parameter (default: 1e-6)

# Returns
- New regularized matrix K + reg*I
"""
function regularize(K::AbstractMatrix, reg::Real = 1.0e-6)
    return K .+ reg .* Matrix(I, size(K, 1), size(K, 2))
end

"""
    solve_kernel_system(K::AbstractMatrix, Y::AbstractArray; solver::SolverMethod=BackslashSolver, kwargs...)

Solve the kernel system K * α = Y using the specified solver method.

# Arguments
- `K::AbstractMatrix`: Kernel matrix (n × n)
- `Y::AbstractArray`: Target values (n × m)
- `solver::SolverMethod`: Solver method to use (default: BackslashSolver)
- `kwargs...`: Additional solver-specific parameters

# Solver-specific kwargs
- `LevenbergMarquardtSolver`: `λ_init=1e-3`, `max_iter=50`

# Returns
- Solution coefficients α such that K * α ≈ Y
"""
function solve_kernel_system(K::AbstractMatrix, Y::AbstractArray; solver::SolverMethod=BackslashSolver, kwargs...)
    if solver == BackslashSolver
        return _solve_backslash(K, Y)
    elseif solver == PinvSolver
        return _solve_pinv(K, Y)
    elseif solver == LevenbergMarquardtSolver
        return _solve_levenberg_marquardt(K, Y; kwargs...)
    else
        error("Unknown solver method: $solver")
    end
end

"""
    _solve_backslash(K::AbstractMatrix, Y::AbstractArray)

Solve using Julia's backslash operator (LU factorization).
"""
function _solve_backslash(K::AbstractMatrix, Y::AbstractArray)
    return K \ Y
end

"""
    _solve_pinv(K::AbstractMatrix, Y::AbstractArray)

Solve using pseudoinverse. Robust for ill-conditioned matrices.
"""
function _solve_pinv(K::AbstractMatrix, Y::AbstractArray)
    return pinv(K) * Y
end

"""
    _solve_levenberg_marquardt(K::AbstractMatrix, Y::AbstractArray; λ_init=1e-3, max_iter=50)

Solve using Levenberg-Marquardt algorithm with adaptive regularization.
This is equivalent to solving (K + λI) \\ Y with adaptive λ.

# Arguments
- `K::AbstractMatrix`: System matrix
- `Y::AbstractArray`: Right-hand side
- `λ_init::Real`: Initial damping parameter (default: 1e-3)
- `max_iter::Int`: Maximum iterations (default: 50)
"""
function _solve_levenberg_marquardt(K::AbstractMatrix, Y::AbstractArray; λ_init=1e-3, max_iter=50)
    # For linear case, this reduces to regularized least squares with adaptive λ
    x = pinv(K) * Y  # Initial guess
    λ = λ_init
    
    for iter in 1:max_iter
        residual = K * x - Y
        
        # Try step with current λ
        δx = -((K' * K + λ * I) \ (K' * residual))
        x_new = x + δx
        
        new_residual = K * x_new - Y
        
        # Accept/reject step and adjust λ
        if norm(new_residual) < norm(residual)
            x = x_new
            λ *= 0.3  # Decrease damping
        else
            λ *= 2.0  # Increase damping
        end
        
        if norm(δx) < 1e-15
            break
        end
    end
    
    return x
end


"""
    get_kernel_interpolant(X::AbstractArray, Y::AbstractArray, mykernel::SKernel; 
                          reg::Real=1.0e-6, solver::SolverMethod=BackslashSolver, kwargs...)

Create a kernel interpolant function for regression.

Solves the kernel regression problem: find coefficients α such that
f(x) = Σᵢ αᵢ K(x, xᵢ) ≈ y

# Arguments
- `X::AbstractArray`: Training input points (n × d matrix)
- `Y::AbstractArray`: Training target values (n × m matrix)
- `mykernel::SKernel`: Kernel function to use
- `reg::Real`: Regularization parameter (default: 1e-6)
- `solver::SolverMethod`: Solver method (default: BackslashSolver)
- `kwargs...`: Additional solver-specific parameters

# Returns
- Function that takes new points and returns predictions

# Example
```julia
kernel = Gaussian(1.0)
f = get_kernel_interpolant(X_train, Y_train, kernel; solver=PinvSolver)
Y_pred = f(X_test)
```
"""
function get_kernel_interpolant(X::AbstractArray, Y::AbstractArray, mykernel::SKernel; 
                               reg::Real=1.0e-6, solver::SolverMethod=BackslashSolver, kwargs...)
    K = evalKmatrix(mykernel, X, X)
    K = regularize(K, reg)
    facts = solve_kernel_system(K, Y; solver=solver, kwargs...)
    fxnew(xnew) = evalKmatrix(mykernel, xnew, X) * facts
    return fxnew
end

"""
    get_kernel_derivative_interpolant(X::AbstractArray, Y::AbstractArray, mykernel::SKernel; 
                                     reg::Real=1.0e-6, solver::SolverMethod=BackslashSolver, kwargs...)

Create a kernel derivative interpolant function.

# Arguments
- `X::AbstractArray`: Training input points (n × d matrix)  
- `Y::AbstractArray`: Training target values (n × m matrix)
- `mykernel::SKernel`: Kernel function (must support evalKmatrix_derivative)
- `reg::Real`: Regularization parameter (default: 1e-6)
- `solver::SolverMethod`: Solver method (default: BackslashSolver)
- `kwargs...`: Additional solver-specific parameters

# Returns
- Function that takes new points and returns derivative predictions
"""
function get_kernel_derivative_interpolant(X::AbstractArray, Y::AbstractArray, mykernel::SKernel; 
                                          reg::Real=1.0e-6, solver::SolverMethod=BackslashSolver, kwargs...)
    K = evalKmatrix(mykernel, X, X)
    K = regularize(K, reg)
    facts = solve_kernel_system(K, Y; solver=solver, kwargs...)
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




"""
    marginal_log_likelihood(θ, X::AbstractArray{T}, Y::AbstractArray{R}, KernelType; reg::Real = 1.0e-5) where {T <: Real, R <: Real}

Compute the marginal log-likelihood for kernel regression with given hyperparameters.

This function is useful for hyperparameter optimization via maximum likelihood estimation.
The marginal likelihood is: p(Y|X,θ) = N(0, K + σₙ²I) where K is the kernel matrix.

# Arguments
- `θ`: Vector of hyperparameters (typically [σ] for kernel bandwidth)
- `X::AbstractArray{T}`: Training input points (n × d matrix)
- `Y::AbstractArray{R}`: Training target values (n × m matrix)
- `KernelType`: Kernel constructor (e.g., Gaussian, Imq)
- `reg::Real`: Noise/regularization parameter (default: 1e-5)

# Returns
- Negative log marginal likelihood (for minimization)
"""
function marginal_log_likelihood(θ, X::AbstractArray{T}, Y::AbstractArray{R}, KernelType; reg::Real = 1.0e-5) where {T <: Real, R <: Real}
    σ = θ[1]
    kernel = KernelType(σ)
    K = evalKmatrix(kernel, X, X) .+ reg .* Matrix(I, size(X, 1), size(X, 1))
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
    interpolant = SimpleKernelRegression.get_kernel_interpolant(X, Y, gaussian; reg=reg)
    X_test = rand(5, 2)
    Y_pred = interpolant(X_test)
    @test size(Y_pred) == (5, 1)
    @test all(isfinite, Y_pred)
    imq = Imq(σ)
    interpolant2 = SimpleKernelRegression.get_kernel_interpolant(X, Y, imq; reg=reg)
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
    interpolant = SimpleKernelRegression.get_kernel_derivative_interpolant(X, Y, gaussian; reg=reg)
    X_test = rand(5, 2)
    Y_pred = interpolant(X_test)
    @test size(Y_pred) == (5, 1)
    @test all(isfinite, Y_pred)
    imq = Imq(σ)
    interpolant2 = SimpleKernelRegression.get_kernel_derivative_interpolant(X, Y, imq; reg=reg)
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
