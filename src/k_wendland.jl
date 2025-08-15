"""
    Wendland{T <: Real} <: SKernel

Wendland compactly supported kernel implementation.

Wendland kernels are compactly supported RBF kernels that provide sparse matrices
and automatic bandwidth selection. They are defined as:
```math
K(x,y) = \\max(1 - \\frac{||x-y||}{\\sigma}, 0)^\\ell \\cdot p(\\frac{||x-y||}{\\sigma})
```

where ℓ = ⌊d/2⌋ + k + 1, d is the dimension, k is the smoothness parameter,
and p(r) is a polynomial that ensures C^(2k) smoothness.

# Fields
- `σ::T`: Support radius (bandwidth parameter)
- `k::Int`: Smoothness parameter (0 ≤ k ≤ 4)
- `d::Int`: Dimension parameter

# Example
```julia
kernel = Wendland(1.0, 2, 2)  # k=2, d=2
K = evalKmatrix(kernel, X, Y)
```
"""
@concrete struct Wendland{T <: Real} <: SKernel
    "Support radius (bandwidth parameter)"
    σ::T
    "Smoothness parameter (0 ≤ k ≤ 4)"
    k::Int
    "Dimension parameter"
    d::Int
end

"""
    Wendland(σ::T, k::Int=2, d::Int=2) where {T}

Constructor for Wendland kernel with support radius σ, smoothness k, and dimension d.
"""
function Wendland(σ::T, k::Int = 2, d::Int = 2) where {T}
    if k < 0 || k > 4
        error("Wendland kernel: k must be between 0 and 4, got $k")
    end
    if d < 1
        error("Wendland kernel: d must be at least 1, got $d")
    end
    return Wendland{T}(σ, k, d)
end


"""
    _wendland_polynomial(r, k, l)

Internal function to compute the Wendland polynomial p(r) for given k and l.
"""
function _wendland_polynomial(r, k, l)
    if k == 0
        return 1.0
    elseif k == 1
        return (l + 1) * r + 1.0
    elseif k == 2
        return (l + 3) * (l + 1) * r^2 + 3.0 * (l + 2) * r + 3.0
    elseif k == 3
        return (l + 5) * (l + 3) * (l + 1) * r^3 +
            (45.0 + 6.0 * l * (l + 6)) * r^2 +
            (15.0 * (l + 3)) * r + 15.0
    elseif k == 4
        return (l + 7) * (l + 5) * (l + 3) * (l + 1) * r^4 +
            (5.0 * (l + 4) * (21.0 + 2.0 * l * (8.0 + l))) * r^3 +
            (45.0 * (14.0 + l * (l + 8.0))) * r^2 +
            (105.0 * (l + 4.0)) * r + 105.0
    else
        error("Wendland polynomial: k must be between 0 and 4, got $k")
    end
end

"""
    _evalKmatrix(wendland::Wendland, xy_dist)

Internal function to evaluate Wendland kernel from precomputed distances.
"""
function _evalKmatrix(wendland::Wendland, xy_dist)
    # Compute parameters
    l = floor(Int, wendland.d / 2) + wendland.k + 1
    e = l + wendland.k

    # Compute factorial ratio
    c = factorial(l + 2 * wendland.k) / factorial(l)

    # Scale distances by support radius
    r_scaled = xy_dist ./ wendland.σ

    # Apply compact support condition
    support_mask = r_scaled .< 1.0

    # Initialize result
    result = zeros(eltype(xy_dist), size(xy_dist))

    # Compute kernel values where support > 0
    for i in eachindex(r_scaled)
        if support_mask[i]
            r = r_scaled[i]
            result[i] = (1.0 - r)^e * _wendland_polynomial(r, wendland.k, l) / c
        end
    end

    return result
end

"""
    evalKmatrix(wendland::Wendland, x::AbstractArray, y::AbstractArray)

Evaluate Wendland kernel matrix between points in x and y.

# Arguments
- `wendland::Wendland`: Wendland kernel instance
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Kernel matrix K of size (n, m) where K[i,j] = K(x[i,:], y[j,:])
"""
function evalKmatrix(wendland::Wendland, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2(x, y)
    return _evalKmatrix(wendland, xy_dist)
end

"""
    evalKernel(wendland::Wendland, x)

Evaluate Wendland kernel function for scalar or vector input.

# Arguments
- `wendland::Wendland`: Wendland kernel instance
- `x`: Input value(s)

# Returns
- Kernel evaluation(s) at input point(s)
"""
function evalKernel(wendland::Wendland, x)
    # Compute parameters
    l = floor(Int, wendland.d / 2) + wendland.k + 1
    e = l + wendland.k
    c = factorial(l + 2 * wendland.k) / factorial(l)

    # Scale input by support radius
    r_scaled = abs.(x) ./ wendland.σ

    # Apply compact support and compute kernel
    result = zeros(eltype(x), size(x))

    for i in eachindex(r_scaled)
        if r_scaled[i] < 1.0
            r = r_scaled[i]
            result[i] = (1.0 - r)^e * _wendland_polynomial(r, wendland.k, l) / c
        end
    end

    return result
end

"""
    evalKmatrix_derivative(wendland::Wendland, x::AbstractArray, y::AbstractArray)

Compute the derivative of the Wendland kernel matrix with respect to the distance.

# Arguments
- `wendland::Wendland`: Wendland kernel with parameters σ, k, d
- `x::AbstractArray`: First set of points (n × d matrix)
- `y::AbstractArray`: Second set of points (m × d matrix)

# Returns
- Matrix of size (n, m) containing the kernel derivative values
"""
function evalKmatrix_derivative(wendland::Wendland, x::AbstractArray, y::AbstractArray)
    xy_dist = pDist2(x, y)

    # Compute parameters
    l = floor(Int, wendland.d / 2) + wendland.k + 1
    e = l + wendland.k
    c = factorial(l + 2 * wendland.k) / factorial(l)

    # Scale distances by support radius
    r_scaled = xy_dist ./ wendland.σ

    # Apply compact support condition
    support_mask = r_scaled .< 1.0

    # Initialize result
    result = zeros(eltype(xy_dist), size(xy_dist))

    # Compute derivative values where support > 0
    for i in eachindex(r_scaled)
        if support_mask[i] && r_scaled[i] > 0
            r = r_scaled[i]
            # Derivative of (1-r)^e * p(r) with respect to r
            derivative = -e * (1.0 - r)^(e - 1) * _wendland_polynomial(r, wendland.k, l) / c +
                (1.0 - r)^e * _wendland_polynomial_derivative(r, wendland.k, l) / c
            result[i] = derivative / wendland.σ  # Chain rule: d/dx = d/dr * dr/dx
        end
    end

    return result
end

"""
    _wendland_polynomial_derivative(r, k, l)

Internal function to compute the derivative of the Wendland polynomial.
"""
function _wendland_polynomial_derivative(r, k, l)
    if k == 0
        return 0.0
    elseif k == 1
        return l + 1.0
    elseif k == 2
        return 2.0 * (l + 3) * (l + 1) * r + 3.0 * (l + 2)
    elseif k == 3
        return 3.0 * (l + 5) * (l + 3) * (l + 1) * r^2 +
            2.0 * (45.0 + 6.0 * l * (l + 6)) * r +
            15.0 * (l + 3)
    elseif k == 4
        return 4.0 * (l + 7) * (l + 5) * (l + 3) * (l + 1) * r^3 +
            3.0 * (5.0 * (l + 4) * (21.0 + 2.0 * l * (8.0 + l))) * r^2 +
            2.0 * (45.0 * (14.0 + l * (l + 8.0))) * r +
            105.0 * (l + 4.0)
    else
        error("Wendland polynomial derivative: k must be between 0 and 4, got $k")
    end
end

"""
    auto_bandwidth(X::AbstractArray, k::Int=2, d::Int=2; 
                   coverage_factor::Real=2.0, min_points::Int=10)

Automatically compute optimal bandwidth for Wendland kernel based on data.

# Arguments
- `X::AbstractArray`: Training points (n × d matrix)
- `k::Int`: Smoothness parameter (default: 2)
- `d::Int`: Dimension (default: 2)
- `coverage_factor::Real`: Factor to ensure good coverage (default: 2.0)
- `min_points::Int`: Minimum points to consider (default: 10)

# Returns
- Optimal bandwidth σ for Wendland kernel
"""
function auto_bandwidth(
        X::AbstractArray, k::Int = 2, d::Int = 2;
        coverage_factor::Real = 2.0, min_points::Int = 10
    )
    n, d_actual = size(X)

    if d_actual != d
        @warn "Dimension mismatch: data has $d_actual dimensions, but kernel expects $d"
        d = d_actual
    end

    if n < min_points
        @warn "Too few points ($n) for automatic bandwidth selection, using default"
        return 1.0
    end

    # Compute fill distance (maximum distance to nearest neighbor)
    if n > 1000
        # For large datasets, sample to estimate fill distance
        sample_size = min(1000, n)
        sample_indices = randperm(n)[1:sample_size]
        X_sample = X[sample_indices, :]
    else
        X_sample = X
    end

    # Compute pairwise distances and find minimum for each point
    distances = pDist2(X_sample, X)
    # Set diagonal to infinity to exclude self-distances
    for i in 1:min(size(distances)...)
        distances[i, i] = Inf
    end
    min_distances = minimum(distances, dims = 2)
    fill_distance = maximum(min_distances[isfinite.(min_distances)])

    # Optimal bandwidth: ensure coverage while maintaining sparsity
    # Use coverage_factor to balance between coverage and sparsity
    optimal_σ = coverage_factor * fill_distance

    # Ensure reasonable bounds
    optimal_σ = max(optimal_σ, 0.1 * fill_distance)
    optimal_σ = min(optimal_σ, 10.0 * fill_distance)

    return optimal_σ
end

"""
    Wendland(X::AbstractArray, k::Int=2, d::Int=2; 
             coverage_factor::Real=2.0, min_points::Int=10)

Constructor for Wendland kernel with automatic bandwidth selection.

# Arguments
- `X::AbstractArray`: Training points for bandwidth estimation
- `k::Int`: Smoothness parameter (default: 2)
- `d::Int`: Dimension (default: 2)
- `coverage_factor::Real`: Factor to ensure good coverage (default: 2.0)
- `min_points::Int`: Minimum points to consider (default: 10)

# Returns
- Wendland kernel with automatically selected bandwidth
"""
function Wendland(
        X::AbstractArray, k::Int = 2, d::Int = 2;
        coverage_factor::Real = 2.0, min_points::Int = 10
    )
    σ = auto_bandwidth(X, k, d; coverage_factor = coverage_factor, min_points = min_points)
    return Wendland(σ, k, d)
end

Base.String(wendland::Wendland) = "k_wendland_$(wendland.k)_$(wendland.d)"

"""
    get_support_radius(wendland::Wendland)

Get the support radius of the Wendland kernel.

# Returns
- Support radius σ
"""
get_support_radius(wendland::Wendland) = wendland.σ

"""
    get_smoothness(wendland::Wendland)

Get the smoothness parameter of the Wendland kernel.

# Returns
- Smoothness parameter k
"""
get_smoothness(wendland::Wendland) = wendland.k

"""
    get_dimension(wendland::Wendland)

Get the dimension parameter of the Wendland kernel.

# Returns
- Dimension parameter d
"""
get_dimension(wendland::Wendland) = wendland.d

"""
    is_compactly_supported(wendland::Wendland)

Check if the Wendland kernel is compactly supported.

# Returns
- Always true for Wendland kernels
"""
is_compactly_supported(wendland::Wendland) = true
