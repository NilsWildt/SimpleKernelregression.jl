"""
    Gaussian{T <: Real} <: SKernel

Gaussian (RBF) kernel implementation.

The Gaussian kernel is defined as:
```math
K(x,y) = \\exp\\left(-\\frac{||x-y||^2}{2\\sigma^2}\\right)
```

# Fields
- `σ::T`: Bandwidth parameter controlling kernel width

# Example
```julia
kernel = Gaussian(1.0)
K = evalKmatrix(kernel, X, Y)
```
"""
@concrete struct Gaussian{T <: Real} <: SKernel
    "Bandwidth parameter"
    σ::T
end

"""
    Gaussian(σ::T) where {T}

Constructor for Gaussian kernel with bandwidth σ.
"""
Gaussian(σ::T) where {T} = Gaussian{T}(σ)

"""
    _evalKmatrix(gaussian::Gaussian, xy_dist)

Internal function to evaluate Gaussian kernel from precomputed squared distances.
"""
_evalKmatrix(gaussian::Gaussian, xy_dist) = @fastmath @.  exp(-1.0 / (2.0 .* gaussian.σ^2) * xy_dist)

"""
    evalKmatrix(gaussian::Gaussian, x::AbstractArray, y::AbstractArray)

Evaluate Gaussian kernel matrix between points in x and y.

# Arguments
- `gaussian::Gaussian`: Gaussian kernel instance
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Kernel matrix K of size (n, m) where K[i,j] = K(x[i,:], y[j,:])
"""
function evalKmatrix(gaussian::Gaussian, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    return _evalKmatrix(gaussian, xy_dist)
end

"""
    evalKernel(gaussian::Gaussian, x)

Evaluate Gaussian kernel function for scalar or vector input.

# Arguments
- `gaussian::Gaussian`: Gaussian kernel instance
- `x`: Input value(s)

# Returns
- Kernel evaluation(s) at input point(s)
"""
function evalKernel(gaussian::Gaussian, x)
    return @fastmath  @.  exp.(- x^2 / (2.0 * gaussian.σ^2))
end

Base.String(::Gaussian) = "k_gaussian"

"""
    evalKmatrix_derivative(gaussian::Gaussian, x::AbstractArray, y::AbstractArray)

Compute the derivative of the Gaussian kernel matrix with respect to the squared distance.

For the Gaussian kernel K(x,y) = exp(-||x-y||²/(2σ²)), the derivative with respect to ||x-y||² is:
dK/d(||x-y||²) = -1/(2σ²) * K(x,y)

# Arguments
- `gaussian::Gaussian`: Gaussian kernel with bandwidth parameter σ
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing the kernel derivative values
"""
function evalKmatrix_derivative(gaussian::Gaussian, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    K = _evalKmatrix(gaussian, xy_dist)
    return @. (-1.0 / (2.0 * gaussian.σ^2)) * K
end
