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
_evalKmatrix(mq::Mq, xy_dist) = (ones(size(xy_dist)) .+ mq.σ .^ 2 .* xy_dist) .^ (0.5)
function evalKmatrix(mq::Mq, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    return K = _evalKmatrix(mq, xy_dist) # Do the inner product and get the polynomial Kernel Matrix
end


function evalKernel(mq::Mq, x)
    return @.  (1.0 .+ mq.σ .^ (-2) .* x .^ 2) .^ (0.5)
end

Base.String(::Mq) = "k_Mq"

"""
    evalKmatrix_derivative(mq::Mq, x::AbstractArray, y::AbstractArray)

Compute the derivative of the Multiquadratic (MQ) kernel matrix.

For the MQ kernel K(x,y) = (1 + σ² * ||x-y||²)^(1/2), the derivative with respect to ||x-y||² is:
dK/d(||x-y||²) = (σ² / 2) * (1 + σ² * ||x-y||²)^(-1/2)

# Arguments
- `mq::Mq`: MQ kernel with scaling parameter σ
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing the kernel derivative values
"""
function evalKmatrix_derivative(mq::Mq, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    return @. (mq.σ^2 / 2.0) * (1.0 + mq.σ^2 * xy_dist)^(-0.5)
end
