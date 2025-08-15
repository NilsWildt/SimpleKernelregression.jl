"""
    Linear <: SKernel

Linear kernel implementation.

The linear kernel is defined as:
```math
K(x,y) = \\langle x,y \\rangle
```
or with bias term:
```math
K(x,y) = \\langle [1;x],[1;y] \\rangle
```

# Fields
- `calc_bias::Bool`: Whether to include bias term (prepend 1 to feature vectors)

# Example
```julia
kernel = Linear(true)   # with bias
kernel = Linear(false)  # without bias
```
"""
@concrete struct Linear <: SKernel
    "Whether to include bias term"
    calc_bias::Bool
end


_evalKmatrix(linear::Linear, xTy) = xTy

function evalKmatrix(linear::Linear, x::AbstractArray, y::AbstractArray)
    if linear.calc_bias
        x = prepend_one(x)
        y = prepend_one(y)
    end
    return K = _evalKmatrix(linear, kernel_dot(x, y)) # Do the inner product and get the polynomial Kernel Matrix
end

function Base.String(linear::Linear)
    if linear.calc_bias == true
        return "k_linear"
    else
        return "k_affine"
    end
end


function evalKernel(linear::Linear, x)
    throw("[-]\nNot implemented.")
end

"""
    evalKmatrix_derivative(linear::Linear, x::AbstractArray, y::AbstractArray)

Compute the derivative of the Linear kernel matrix.

For the Linear kernel K(x,y) = <x,y> (or <[1;x],[1;y]> with bias), 
the derivative with respect to the dot product is simply 1.

# Arguments
- `linear::Linear`: Linear kernel
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Matrix of ones of size (n, m) since dK/d(<x,y>) = 1
"""
function evalKmatrix_derivative(linear::Linear, x::AbstractArray, y::AbstractArray)
    return ones(size(x, 1), size(y, 1))
end
