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


_evalKmatrix(epanechnikov::Epanechnikov, xy_dist) = @fastmath @. 3/4 * (1 - (xy_dist / epanechnikov.h)^2) * float(xy_dist <= epanechnikov.h)

function evalKmatrix(epanechnikov::Epanechnikov, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2(x, y)  # Use actual distances, not squared
    return K = _evalKmatrix(epanechnikov, xy_dist)
end

function evalKernel(epanechnikov::Epanechnikov, x)
    return @. 3/4 * (1 - x^2 / (epanechnikov.h^2)) * float(abs(x) <= epanechnikov.h)
end

Base.String(::Epanechnikov) = "k_epanechnikov"
