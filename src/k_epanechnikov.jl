# Define datatypes for kernels
""" 
Epanechnikov <: Kernel
Epanechnikov kernel
```math
K(x,y) = 
```
scaling factor ``sigma``.
"""

mutable struct Epanechnikov <: SKernel
    "Scaling"
    σ::Float64 # Überbleibsel
end

# Constructor of the struct above.
# Epanechnikov(σ::T) where T = Epanechnikov{T}(σ, [])

_evalKmatrix(epanechnikov::Epanechnikov, xy_dist) = @fastmath @. epanechnikov.σ .* 3 / 4 * (1 - xy_dist) * float(abs(xy_dist) <= 1.0)

function evalKmatrix(epanechnikov::Epanechnikov, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    return K = _evalKmatrix(epanechnikov, xy_dist) # Do the inner product and get the polynomial Kernel Matrix
end

function evalKernel(epanechnikov::Epanechnikov, x)
    return epanechnikov.σ .* (3 / 4 .* (1 .- x) .* float(abs.(x) .<= 1.0))
end

Base.String(::Epanechnikov) = "k_epanechnikov"
