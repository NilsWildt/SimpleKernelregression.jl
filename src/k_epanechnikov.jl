# Define datatypes for kernels
""" 
Epanechnikov <: SKernel
Epanechnikov kernel
```math
K(x,y) = \\frac{3}{4} \\left(1 - \\frac{\\|x-y\\|^2}{h^2}\\right) \\mathbf{1}_{\\{\\|x-y\\| \\leq h\\}}
```
where `h` is the bandwidth parameter.
"""

@concrete mutable struct Epanechnikov <: SKernel
    "Bandwidth parameter"
    h::Float64
end


_evalKmatrix(epanechnikov::Epanechnikov, xy_dist) = @. 3 / 4 * (1 - (xy_dist / epanechnikov.h)^2) * float(xy_dist <= epanechnikov.h)

function evalKmatrix(epanechnikov::Epanechnikov, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2(x, y)  # Use actual distances, not squared
    return K = _evalKmatrix(epanechnikov, xy_dist)
end

function evalKernel(epanechnikov::Epanechnikov, x)
    return @. 3 / 4 * (1 - x^2 / (epanechnikov.h^2)) * float(abs(x) <= epanechnikov.h)
end

Base.String(::Epanechnikov) = "k_epanechnikov"

"""
    evalKmatrix_derivative(epanechnikov::Epanechnikov, x::AbstractArray, y::AbstractArray)

Compute the derivative of the Epanechnikov kernel matrix.

For the Epanechnikov kernel K(x,y) = (3/4) * (1 - (||x-y||/h)²) * I(||x-y|| ≤ h), 
the derivative with respect to ||x-y|| is:
dK/d(||x-y||) = -(3/2h²) * (||x-y||/h) * I(||x-y|| ≤ h)

# Arguments
- `epanechnikov::Epanechnikov`: Epanechnikov kernel with bandwidth parameter h
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing the kernel derivative values
"""
function evalKmatrix_derivative(epanechnikov::Epanechnikov, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2(x, y)  # Use actual distances, not squared
    return @. (-3.0 / (2.0 * epanechnikov.h^2)) * (xy_dist / epanechnikov.h) * float(xy_dist <= epanechnikov.h)
end
