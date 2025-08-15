# Define datatypes for kernels
""" 
k_Linear <: Kernel
Linear kernel
```math
K(x,y) = (x,y)
```
(.,.) is the inner product. No offset --> affine.
"""
@concrete struct Linear <: SKernel
    calc_bias::Bool
end


_evalKmatrix(linear::Linear, xTy) =  xTy

function evalKmatrix(linear::Linear, x::AbstractArray, y::AbstractArray)
    if linear.calc_bias
        x = prepend_one(x)
    end
  K = _evalKmatrix(linear, kernel_dot(x, y)) # Do the inner product and get the polynomial Kernel Matrix
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
