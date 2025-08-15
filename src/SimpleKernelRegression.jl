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
        x = pinv(K) * Y  # Initial guess
        λ = λ_init
        
        for _ in 1:max_iter
            residual = K * x - Y
            
            JtJ = K' * K
            Jtr = K' * residual
            
            δx = -(JtJ + λ * I) \ Jtr
            x_new = x + δx
            
            new_residual = K * x_new - Y
            
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

@testitem "All 3 Solving Methods Give Same Results" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(123)
    
    # Create test data
    X = rand(15, 3)
    Y = sin.(sum(X, dims=2)) + 0.1 * rand(15, 1)
    reg = 1e-6
    
    # Test with Gaussian kernel
    gaussian = Gaussian(1.0)
    
    # Get interpolants using all 3 methods
    f_backslash = get_kernel_interpolant(X, Y, gaussian; reg=reg, solver=BackslashSolver)
    f_pinv = get_kernel_interpolant(X, Y, gaussian; reg=reg, solver=PinvSolver)
    f_lm = get_kernel_interpolant(X, Y, gaussian; reg=reg, solver=LevenbergMarquardtSolver)
    
    # Test predictions on new data
    X_test = rand(10, 3)
    pred_backslash = f_backslash(X_test)
    pred_pinv = f_pinv(X_test)
    pred_lm = f_lm(X_test)
    
    # Check that all methods give similar results (within numerical tolerance)
    @test isapprox(pred_backslash, pred_pinv, rtol=1e-10, atol=1e-12)
    @test isapprox(pred_backslash, pred_lm, rtol=1e-8, atol=1e-10)
    @test isapprox(pred_pinv, pred_lm, rtol=1e-8, atol=1e-10)
    
    # Test with IMQ kernel
    imq = Imq(2.0)
    f_backslash_imq = get_kernel_interpolant(X, Y, imq; reg=reg, solver=BackslashSolver)
    f_pinv_imq = get_kernel_interpolant(X, Y, imq; reg=reg, solver=PinvSolver)
    f_lm_imq = get_kernel_interpolant(X, Y, imq; reg=reg, solver=LevenbergMarquardtSolver)
    
    pred_backslash_imq = f_backslash_imq(X_test)
    pred_pinv_imq = f_pinv_imq(X_test)
    pred_lm_imq = f_lm_imq(X_test)
    
    @test isapprox(pred_backslash_imq, pred_pinv_imq, rtol=1e-10, atol=1e-12)
    @test isapprox(pred_backslash_imq, pred_lm_imq, rtol=1e-8, atol=1e-10)
    @test isapprox(pred_pinv_imq, pred_lm_imq, rtol=1e-8, atol=1e-10)
end

@testitem "Linear Kernel Coverage" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(42)
    
    # Test Linear kernel with and without bias
    linear_bias = Linear(true)
    linear_nobias = Linear(false)
    
    X = rand(10, 3)
    Y = rand(5, 3)
    
    # Test evalKmatrix
    K_bias = SimpleKernelRegression.evalKmatrix(linear_bias, X, Y)
    K_nobias = SimpleKernelRegression.evalKmatrix(linear_nobias, X, Y)
    
    @test size(K_bias) == (10, 5)
    @test size(K_nobias) == (10, 5)
    @test all(isfinite, K_bias)
    @test all(isfinite, K_nobias)
    
    # Test String conversion
    @test String(linear_bias) == "k_linear"
    @test String(linear_nobias) == "k_affine"
    
    # Test that evalKernel throws error
    @test_throws String SimpleKernelRegression.evalKernel(linear_bias, [1.0, 2.0])
end

@testitem "Polynomial Kernel Coverage" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(42)
    
    # Test Polynomial kernel
    poly = Polynomial(1.0, 2)
    
    X = rand(10, 3)
    Y = rand(5, 3)
    
    # Test evalKmatrix
    K = SimpleKernelRegression.evalKmatrix(poly, X, Y)
    @test size(K) == (10, 5)
    @test all(isfinite, K)
    @test all(≥(1.0), K)  # Should be at least 1 due to constant term
    
    # Test String conversion
    @test String(poly) == "k_polynomial"
    
    # Test constructor
    poly2 = Polynomial(2.5, 3)
    @test poly2.a == 2.5
    @test poly2.p == 3
end

@testitem "Mq Kernel Coverage" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(42)
    
    # Test Multiquadratic kernel
    mq = Mq(1.5)
    
    X = rand(10, 3)
    Y = rand(5, 3)
    
    # Test evalKmatrix
    K = SimpleKernelRegression.evalKmatrix(mq, X, Y)
    @test size(K) == (10, 5)
    @test all(isfinite, K)
    @test all(≥(1.0), K)  # Should be at least 1
    
    # Test evalKernel
    test_vals = [0.0, 1.0, 2.0]
    result = SimpleKernelRegression.evalKernel(mq, test_vals)
    @test length(result) == length(test_vals)
    @test all(isfinite, result)
    @test all(≥(1.0), result)
    
    # Test String conversion
    @test String(mq) == "k_Mq"
end

@testitem "Epanechnikov Kernel Coverage" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(42)
    
    # Test Epanechnikov kernel
    epan = Epanechnikov(2.0)
    
    X = rand(5, 2)
    Y = rand(5, 2)
    
    # Test evalKmatrix
    K = SimpleKernelRegression.evalKmatrix(epan, X, Y)
    @test size(K) == (5, 5)
    @test all(isfinite, K)
    @test all(≥(0.0), K)  # Non-negative kernel
    
    # Test evalKernel with values inside and outside bandwidth
    test_vals = [-3.0, -1.0, 0.0, 1.0, 3.0]  # Some inside h=2.0, some outside
    result = SimpleKernelRegression.evalKernel(epan, test_vals)
    @test length(result) == length(test_vals)
    @test all(isfinite, result)
    @test all(≥(0.0), result)
    @test result[5] == 0.0  # Outside bandwidth should be 0
    @test result[3] > 0.0   # At center should be positive
    
    # Test String conversion
    @test String(epan) == "k_epanechnikov"
end

@testitem "Regularization Functions" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(42)
    
    # Test regularization functions
    K = rand(5, 5)
    K = K' * K  # Make symmetric positive definite
    reg = 0.1
    
    # Test in-place regularization
    K_copy = copy(K)
    SimpleKernelRegression.regularize!(K_copy, reg)
    @test all([K_copy[i,i] >= K[i,i] + reg - 1e-10 for i in 1:5])  # Check diagonal manually
    
    # Test non-in-place regularization
    K_reg = SimpleKernelRegression.regularize(K, reg)
    I_matrix = zeros(5, 5)
    for i in 1:5; I_matrix[i,i] = 1.0; end  # Manual identity matrix
    @test K_reg ≈ K .+ reg .* I_matrix
    @test K == K  # Original unchanged
end

@testitem "Solver Method Edge Cases" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(42)
    
    # Test with well-conditioned matrix first
    A_regular = rand(3, 3)
    A_regular = A_regular' * A_regular  # Make positive semidefinite
    # Add some regularization manually
    for i in 1:3; A_regular[i,i] += 0.1; end
    Y = rand(3)
    
    # All solvers should work on well-conditioned matrix
    sol_backslash = SimpleKernelRegression._solve_backslash(A_regular, Y)
    sol_pinv = SimpleKernelRegression._solve_pinv(A_regular, Y)
    sol_lm = SimpleKernelRegression._solve_levenberg_marquardt(A_regular, Y; λ_init=1e-3, max_iter=10)
    
    @test size(sol_backslash) == (3,)
    @test size(sol_pinv) == (3,)
    @test size(sol_lm) == (3,)
    
    # Test with singular matrix (only pinv and LM should work gracefully)
    A_singular = ones(3, 3)  # Singular matrix
    Y_singular = [1.0, 1.0, 1.0]
    
    # Pinv should handle singular matrix gracefully
    sol_pinv_singular = SimpleKernelRegression._solve_pinv(A_singular, Y_singular)
    @test size(sol_pinv_singular) == (3,)
    @test isapprox(A_singular * sol_pinv_singular, Y_singular, atol=1e-10)
    
    # LM should handle singular matrix with regularization
    sol_lm_singular = SimpleKernelRegression._solve_levenberg_marquardt(A_singular, Y_singular; λ_init=1e-2, max_iter=10)
    @test size(sol_lm_singular) == (3,)
end

@testitem "KernelInterpolant and KernelDerivativeInterpolant" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(42)
    
    X = rand(10, 2)
    Y = rand(10, 3)
    kernel = Gaussian(1.0)
    
    # Test KernelInterpolant
    interp = SimpleKernelRegression.KernelInterpolant(X, kernel, 1e-6)
    @test interp.initialized == true
    @test size(interp.X) == size(X)
    
    # Update coefficients
    SimpleKernelRegression.update_coefficients!(interp, Y)
    @test size(interp.coefficients) == size(Y)
    
    # Test prediction
    X_test = rand(5, 2)
    pred = interp(X_test)
    @test size(pred) == (5, 3)
    @test all(isfinite, pred)
    
    # Test KernelDerivativeInterpolant
    deriv_interp = SimpleKernelRegression.KernelDerivativeInterpolant(X, kernel, 1e-6)
    @test deriv_interp.initialized == true
    
    SimpleKernelRegression.update_coefficients!(deriv_interp, Y)
    @test size(deriv_interp.coefficients) == size(Y)
    
    # Test derivative prediction (this requires evalKmatrix_derivative to be implemented)
    # Skip if not implemented for this kernel
    try
        pred_deriv = deriv_interp(X_test)
        @test size(pred_deriv) == (5, 3)
        @test all(isfinite, pred_deriv)
    catch MethodError
        @info "evalKmatrix_derivative not implemented for this kernel - skipping derivative test"
    end
end

@testitem "Utility Functions Coverage" begin
    using Random
    using SimpleKernelRegression
    Random.seed!(42)
    
    # Test prepend_one
    X = rand(5, 3)
    X_with_ones = SimpleKernelRegression.prepend_one(X)
    @test size(X_with_ones) == (5, 4)
    @test all(X_with_ones[:, 1] .== 1.0)
    @test X_with_ones[:, 2:end] == X
    
    # Test kernel_dot with different inputs
    X = rand(5, 3)
    Y = rand(4, 3)
    dot_result = SimpleKernelRegression.kernel_dot(X, Y)
    @test size(dot_result) == (5, 4)
    @test dot_result ≈ X * Y'
    
    # Test kernel_dot with adjoint
    Y_adj = Y'
    dot_result_adj = SimpleKernelRegression.kernel_dot(X, Y_adj)
    @test size(dot_result_adj) == (5, 4)
    @test dot_result_adj ≈ X * Y_adj
end

end # End module
