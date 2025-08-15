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
_evalKmatrix(imq::Imq, xy_dist) = @fastmath (ones(size(xy_dist)) .+ imq.σ .^ (-2) .* xy_dist) .^ (-0.5)
function evalKmatrix(imq::Imq, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    return K = _evalKmatrix(imq, xy_dist) # Do the inner product and get the polynomial Kernel Matrix
end

function evalKernel(imq::Imq, x)
    return @fastmath  @. (1.0 .+ imq.σ .^ (-2) .* x .^ 2) .^ (-0.5)
end


Base.String(::Imq) = "k_imq"

"""
    evalKmatrix_derivative(imq::Imq, x::AbstractArray, y::AbstractArray)

Compute the derivative of the Inverse Multiquadratic (IMQ) kernel matrix.

For the IMQ kernel K(x,y) = (1 + ||x-y||²/σ²)^(-1/2), the derivative with respect to ||x-y||² is:
dK/d(||x-y||²) = -1/(2σ²) * (1 + ||x-y||²/σ²)^(-3/2)

# Arguments
- `imq::Imq`: IMQ kernel with scaling parameter σ
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing the kernel derivative values
"""
function evalKmatrix_derivative(imq::Imq, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2Squared(x, y)
    # ϕ(r) = (1 + (r/σ)^2)^(-1/2); derivative wrt r^2 scales similarly for matrix form
    dϕ(r) = -0.5 * (1 + (r / imq.σ)^2)^(-1.5) * (1 / imq.σ^2)
    return dϕ.(xy_dist)
end
