# Define datatypes for kernels
""" 
k_mq <: Kernel
Multiquadratic kernel
```math
K(x,y) = sqrt()
```
scaling factor ``sigma``.
"""

@concrete struct Mq{T <: Real} <: SKernel
    "Scaling"
    σ::T
end

# Constructor of the struct above.
Mq(σ::T) where {T} = Mq{T}(σ)
_evalKmatrix(mq::Mq, xy_dist) = @fastmath (ones(size(xy_dist)) .+ mq.σ .^ 2 .* xy_dist) .^ (0.5)
function evalKmatrix(mq::Mq, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    return K = _evalKmatrix(mq, xy_dist) # Do the inner product and get the polynomial Kernel Matrix
end


function evalKernel(mq::Mq, x)
    return @fastmath  @.  (1.0 .+ mq.σ .^ (-2) .* x .^ 2) .^ (0.5)
end

Base.String(::Mq) = "k_Mq"
