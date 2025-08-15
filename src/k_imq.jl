# Define datatypes for kernels
""" 
k_imq <: Kernel
Inverse multiquadratic kernel
```math
K(x,y) = sqrt()
```
scaling factor ``sigma``.
"""

@concrete struct Imq{T <: Real} <: SKernel
    "Scaling"
    σ::T
end

# Constructor of the struct above.
Imq(σ::T) where {T} = Imq{T}(σ)
_evalKmatrix(imq::Imq, xy_dist) = @fastmath (ones(size(xy_dist)) .+ imq.σ.^(-2) .* xy_dist).^(-0.5)
function evalKmatrix(imq::Imq, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    K = _evalKmatrix(imq, xy_dist) # Do the inner product and get the polynomial Kernel Matrix
end

function evalKernel(imq::Imq, x)
    return @fastmath  @. (1.0 .+ imq.σ.^(-2) .* x.^2).^(-0.5)
end


Base.String(::Imq) = "k_imq"
