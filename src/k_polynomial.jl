# Define datatypes for kernels
""" 
k_polynomial <: Kernel
Polynomial kernel
```math
K(x,y) = ((x,y) + a)^p
```
(.,.) inner product, additive constant ``a``, and degree ``p``.
"""

@concrete struct Polynomial{T <: Real} <: SKernel
    "Constant"
    a::T
    "Polynomial degree"
    p::Int64
end

# Constructor of the struct above.
Polynomial(a::T, p::Int) where {T} = Polynomial{T}(a, p)
_evalKmatrix(polynomial::Polynomial, xTy) = (xTy .+ polynomial.a) .^ polynomial.p
function evalKmatrix(polynomial::Polynomial, x::AbstractArray, y::AbstractArray)
    return K = _evalKmatrix(polynomial, kernel_dot(x, y')) # Do the inner product and get the polynomial Kernel Matrix
end


Base.String(::Polynomial) = "k_polynomial"

function evalKernel(polynomial::Polynomial, x)
    return @. (x + polynomial.a)^polynomial.p
end

"""
    evalKmatrix_derivative(polynomial::Polynomial, x::AbstractArray, y::AbstractArray)

Compute the derivative of the Polynomial kernel matrix with respect to the dot product.

For the Polynomial kernel K(x,y) = (<x,y> + a)^p, the derivative with respect to <x,y> is:
dK/d(<x,y>) = p * (<x,y> + a)^(p-1)

# Arguments
- `polynomial::Polynomial`: Polynomial kernel with constant a and degree p
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing the kernel derivative values
"""
function evalKmatrix_derivative(polynomial::Polynomial, x::AbstractArray, y::AbstractArray)
    if polynomial.p == 0
        return zeros(size(x, 1), size(y, 1))  # Derivative of constant is 0
    end
    xTy = kernel_dot(x, y')
    return @. polynomial.p * (xTy + polynomial.a)^(polynomial.p - 1)
end
