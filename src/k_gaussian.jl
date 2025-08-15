# Define datatypes for kernels
""" 
k_Gaussian <: Kernel
Gaussian kernel
```math
K(x,y) = exp(-1/sigma^2 cdot ||x-y||^2)
```
scaling factor ``sigma``.
"""

@concrete struct Gaussian{T <: Real} <: SKernel
    "Scaling"
    σ::T
end

# Constructor of the struct above.
Gaussian(σ::T) where {T} = Gaussian{T}(σ)

_evalKmatrix(gaussian::Gaussian, xy_dist) = @fastmath @.  exp(-1.0 / (2.0 .* gaussian.σ^2) * xy_dist)

function evalKmatrix(gaussian::Gaussian, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    return K = _evalKmatrix(gaussian, xy_dist) # Do the inner product and get the polynomial Kernel Matrix
end

function evalKernel(gaussian::Gaussian, x)
    return @fastmath  @.  exp.(- x^2 / (2.0 * gaussian.σ^2)) # 1. ./ (sqrt(2.0 * pi) * gaussian.σ) .*
end

Base.String(::Gaussian) = "k_gaussian"
