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
_evalKmatrix(polynomial::Polynomial, xTy) = @fastmath (xTy .+ polynomial.a) .^ polynomial.p
function evalKmatrix(polynomial::Polynomial, x::AbstractArray, y::AbstractArray)
    return K = _evalKmatrix(polynomial, kernel_dot(x, y')) # Do the inner product and get the polynomial Kernel Matrix
end


Base.String(::Polynomial) = "k_polynomial"
