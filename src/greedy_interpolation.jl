"""
    GreedyKernelInterpolant{T, KT <: SKernel}

Greedy kernel interpolation using power function for adaptive point selection.

This implementation follows the VKOGA (Vectorial Kernel Orthogonal Greedy Algorithm)
approach, which iteratively selects the most informative points using the power function
and constructs an orthogonal basis for efficient interpolation.

# Fields
- `kernel::KT`: Kernel function
- `centers::AbstractArray{T}`: Selected interpolation centers
- `coefficients::AbstractArray{T}`: Interpolation coefficients
- `orthogonal_basis::AbstractArray{T}`: Orthogonal basis matrix
- `power_history::Vector{T}`: History of power function values
- `error_history::Vector{T}`: History of approximation errors
- `tolerance::T`: Convergence tolerance
- `max_iterations::Int`: Maximum number of iterations
- `greedy_type::Symbol`: Type of greedy selection ('p_greedy', 'f_greedy', 'fp_greedy')
"""
@concrete mutable struct GreedyKernelInterpolant{T, KT <: SKernel}
    "Kernel function"
    kernel::KT
    "Selected interpolation centers"
    centers::AbstractArray{T}
    "Interpolation coefficients"
    coefficients::AbstractArray{T}
    "Orthogonal basis matrix (Newton basis)"
    orthogonal_basis::AbstractArray{T}
    "History of power function values"
    power_history::Vector{T}
    "History of approximation errors"
    error_history::Vector{T}
    "Convergence tolerance"
    tolerance::T
    "Maximum number of iterations"
    max_iterations::Int
    "Type of greedy selection"
    greedy_type::Symbol
    "Regularization parameter"
    regularization::T
    "Verbose output"
    verbose::Bool
    "Training history"
    training_history::Dict{String, Any}
end

"""
    GreedyKernelInterpolant(kernel::SKernel; 
                            tolerance::Real=1e-10, 
                            max_iterations::Int=1000,
                            greedy_type::Symbol=:p_greedy,
                            regularization::Real=1e-12,
                            verbose::Bool=true)

Construct a greedy kernel interpolant.

# Arguments
- `kernel::SKernel`: Kernel function to use
- `tolerance::Real`: Convergence tolerance (default: 1e-10)
- `max_iterations::Int`: Maximum number of iterations (default: 1000)
- `greedy_type::Symbol`: Type of greedy selection (default: :p_greedy)
  - `:p_greedy`: Power function greedy (selects points with maximum power function)
  - `:f_greedy`: Function value greedy (selects points with maximum residual)
  - `:fp_greedy`: Combined greedy (selects points with maximum f/p ratio)
- `regularization::Real`: Regularization parameter (default: 1e-12)
- `verbose::Bool`: Verbose output (default: true)

# Returns
- GreedyKernelInterpolant instance
"""
function GreedyKernelInterpolant(
        kernel::SKernel;
        tolerance::Real = 1.0e-10,
        max_iterations::Int = 1000,
        greedy_type::Symbol = :p_greedy,
        regularization::Real = 1.0e-12,
        verbose::Bool = true
    )

    if !(greedy_type in [:p_greedy, :f_greedy, :fp_greedy])
        error("Invalid greedy_type: $greedy_type. Must be one of [:p_greedy, :f_greedy, :fp_greedy]")
    end

    T = promote_type(typeof(tolerance), typeof(regularization))

    return GreedyKernelInterpolant{T, typeof(kernel)}(
        kernel,
        Array{T}(undef, 0, 0),
        Array{T}(undef, 0, 0),
        Array{T}(undef, 0, 0),
        T[],
        T[],
        T(tolerance),
        max_iterations,
        greedy_type,
        T(regularization),
        verbose,
        Dict{String, Any}()
    )
end

"""
    fit!(interpolant::GreedyKernelInterpolant, X::AbstractArray, Y::AbstractArray;
         X_val::AbstractArray=nothing, Y_val::AbstractArray=nothing)

Fit the greedy kernel interpolant to the data.

# Arguments
- `interpolant::GreedyKernelInterpolant`: The interpolant to fit
- `X::AbstractArray`: Training input points (n × d matrix)
- `Y::AbstractArray`: Training target values (n × m matrix)
- `X_val::AbstractArray`: Validation input points (optional)
- `Y_val::AbstractArray`: Validation target values (optional)

# Returns
- The fitted interpolant
"""
function fit!(
        interpolant::GreedyKernelInterpolant, X::AbstractArray, Y::AbstractArray;
        X_val = nothing, Y_val = nothing
    )

    T = eltype(X)
    n, d = size(X)
    m = size(Y, 2)

    # Initialize
    interpolant.centers = Array{T}(undef, 0, d)
    interpolant.coefficients = Array{T}(undef, 0, m)
    interpolant.orthogonal_basis = Array{T}(undef, 0, 0)
    interpolant.power_history = T[]
    interpolant.error_history = T[]

    # Initialize training history
    interpolant.training_history = Dict(
        "n" => Int[],
        "p" => T[],
        "f" => T[],
        "p_selected" => T[],
        "p_val" => T[],
        "f_val" => T[]
    )

    # Check if validation data is provided
    has_validation = X_val !== nothing && Y_val !== nothing

    # Initialize residual
    residual = copy(Y)

    # Initialize power function values
    power_vals = ones(T, n)

    # Main greedy loop
    if interpolant.verbose
        println("Starting greedy kernel interpolation...")
        println("  Greedy type: $(interpolant.greedy_type)")
        println("  Tolerance: $(interpolant.tolerance)")
        println("  Max iterations: $(interpolant.max_iterations)")
        println("  Training points: $n")
        println("  Output dimension: $m")
    end

    for iteration in 1:interpolant.max_iterations

        # Selection rule based on greedy type
        if interpolant.greedy_type == :p_greedy
            idx = argmax(power_vals)
            f_max = maximum(sqrt.(sum(residual .^ 2, dims = 2)))
            p_max = power_vals[idx]
        elseif interpolant.greedy_type == :f_greedy
            f_squared = sum(residual .^ 2, dims = 2)
            idx = argmax(f_squared)
            f_max = sqrt(f_squared[idx])
            p_max = power_vals[idx]
        elseif interpolant.greedy_type == :fp_greedy
            f_squared = sum(residual .^ 2, dims = 2)
            # Avoid division by zero
            safe_power = max.(power_vals, 1.0e-12)
            ratio = f_squared ./ safe_power
            idx = argmax(ratio)
            f_max = sqrt(f_squared[idx])
            p_max = power_vals[idx]
        end

        # Ensure idx is a scalar row index
        if idx isa CartesianIndex
            idx = idx[1]
        end

        # Check convergence
        if f_max <= interpolant.tolerance
            if interpolant.verbose
                println("Converged: residual error $(f_max) <= tolerance $(interpolant.tolerance)")
            end
            break
        end

        if p_max <= interpolant.tolerance
            if interpolant.verbose
                println("Converged: power function $(p_max) <= tolerance $(interpolant.tolerance)")
            end
            break
        end

        # Select the new center - ensure 2D structure
        new_center = reshape(X[idx, :], 1, size(X, 2))

        # Update centers - ensure proper dimensions
        if isempty(interpolant.centers)
            interpolant.centers = copy(new_center)
        else
            # Ensure we maintain 2D structure - new_center is already properly shaped
            interpolant.centers = vcat(interpolant.centers, new_center)
        end


        # Compute kernel values for the new center
        if size(interpolant.centers, 1) == 1
            # First center
            K_new = evalKmatrix(interpolant.kernel, new_center, new_center)
            K_new[1, 1] += interpolant.regularization  # Add regularization to diagonal

            # Initialize orthogonal basis
            interpolant.orthogonal_basis = reshape([1.0 / sqrt(K_new[1, 1])], 1, 1)

            # Compute coefficient
            coeff = residual[idx, :] / sqrt(K_new[1, 1])
            interpolant.coefficients = reshape(coeff, 1, m)

        else
            # Subsequent centers
            n_centers = size(interpolant.centers, 1)

            # Compute kernel values
            K_new_old = evalKmatrix(interpolant.kernel, new_center, interpolant.centers[1:(end - 1), :])
            K_new_new = evalKmatrix(interpolant.kernel, new_center, new_center)
            K_new_new[1, 1] += interpolant.regularization  # Add regularization to diagonal

            # Compute orthogonal basis for new center
            V_new = zeros(T, n_centers)
            V_new[1:(end - 1)] = K_new_old * interpolant.orthogonal_basis'

            # Ensure numerical stability for the orthogonal component
            orthogonal_component = K_new_new[1, 1] - sum(V_new[1:(end - 1)] .^ 2)
            V_new[end] = sqrt(max(orthogonal_component, interpolant.regularization))

            # Update orthogonal basis
            new_row = zeros(T, 1, n_centers)
            new_row[1, 1:(end - 1)] = -V_new[1:(end - 1)]' * interpolant.orthogonal_basis
            new_row[1, end] = V_new[end]

            interpolant.orthogonal_basis = vcat(
                hcat(interpolant.orthogonal_basis, zeros(T, n_centers - 1, 1)),
                new_row
            )

            # Compute coefficient
            coeff = residual[idx, :] / V_new[end]
            interpolant.coefficients = vcat(interpolant.coefficients, coeff')
        end

        # Update residual
        prediction = predict(interpolant, X)
        residual = Y - prediction

        # Check for numerical instability
        if !all(isfinite, residual)
            if interpolant.verbose
                println("Warning: Numerical instability detected at iteration $iteration")
            end
            # Reset to previous state and stop
            break
        end

        # Update power function values
        power_vals = power_function(interpolant.kernel, interpolant.centers, X, interpolant.regularization)

        # Store history
        push!(interpolant.training_history["n"], iteration)
        push!(interpolant.training_history["p"], p_max)
        push!(interpolant.training_history["f"], f_max)
        push!(interpolant.training_history["p_selected"], p_max)
        push!(interpolant.power_history, p_max)
        push!(interpolant.error_history, f_max)

        # Validation tracking
        if has_validation
            p_val = maximum(power_function(interpolant.kernel, interpolant.centers, X_val, interpolant.regularization))
            f_val = maximum(sqrt.(sum((Y_val - predict(interpolant, X_val)) .^ 2, dims = 2)))
            push!(interpolant.training_history["p_val"], p_val)
            push!(interpolant.training_history["f_val"], f_val)
        end

        # Progress reporting
        if interpolant.verbose && (iteration % 10 == 0 || iteration == 1)
            println("  Iteration $iteration: centers=$(size(interpolant.centers, 1)), f_max=$(round(f_max, sigdigits = 3)), p_max=$(round(p_max, sigdigits = 3))")
        end
    end

    if interpolant.verbose
        println("Training completed with $(size(interpolant.centers, 1)) centers")
        println("Final residual error: $(interpolant.error_history[end])")
        println("Final power function: $(interpolant.power_history[end])")
    end

    return interpolant
end

"""
    fit(interpolant::GreedyKernelInterpolant, X::AbstractArray, Y::AbstractArray;
        X_val::AbstractArray=nothing, Y_val::AbstractArray=nothing)

Fit the greedy kernel interpolant to the data (non-mutating version).

# Arguments
- `interpolant::GreedyKernelInterpolant`: The interpolant to fit
- `X::AbstractArray`: Training input points (n × d matrix)
- `Y::AbstractArray`: Training target values (n × m matrix)
- `X_val::AbstractArray`: Validation input points (optional)
- `Y_val::AbstractArray`: Validation target values (optional)

# Returns
- A new fitted interpolant
"""
function fit(
        interpolant::GreedyKernelInterpolant, X::AbstractArray, Y::AbstractArray;
        X_val = nothing, Y_val = nothing
    )
    new_interpolant = deepcopy(interpolant)
    return fit!(new_interpolant, X, Y; X_val = X_val, Y_val = Y_val)
end

"""
    predict(interpolant::GreedyKernelInterpolant, X::AbstractArray)

Predict using the fitted greedy kernel interpolant.

# Arguments
- `interpolant::GreedyKernelInterpolant`: The fitted interpolant
- `X::AbstractArray`: Input points (n × d matrix)

# Returns
- Predictions (n × m matrix)
"""
function predict(interpolant::GreedyKernelInterpolant, X::AbstractArray)
    if isempty(interpolant.centers)
        return zeros(eltype(X), size(X, 1), 0)
    end

    # Compute kernel values
    K = evalKmatrix(interpolant.kernel, X, interpolant.centers)

    # Apply orthogonal basis transformation
    V = K * interpolant.orthogonal_basis'

    # Compute predictions
    predictions = V * interpolant.coefficients

    return predictions
end

"""
    (interpolant::GreedyKernelInterpolant)(X::AbstractArray)

Callable interface for prediction.

# Arguments
- `X::AbstractArray`: Input points

# Returns
- Predictions
"""
function (interpolant::GreedyKernelInterpolant)(X::AbstractArray)
    return predict(interpolant, X)
end

"""
    get_centers(interpolant::GreedyKernelInterpolant)

Get the selected interpolation centers.

# Returns
- Matrix of centers
"""
get_centers(interpolant::GreedyKernelInterpolant) = interpolant.centers

"""
    get_coefficients(interpolant::GreedyKernelInterpolant)

Get the interpolation coefficients.

# Returns
- Matrix of coefficients
"""
get_coefficients(interpolant::GreedyKernelInterpolant) = interpolant.coefficients

"""
    get_orthogonal_basis(interpolant::GreedyKernelInterpolant)

Get the orthogonal basis matrix.

# Returns
- Orthogonal basis matrix
"""
get_orthogonal_basis(interpolant::GreedyKernelInterpolant) = interpolant.orthogonal_basis

"""
    get_power_history(interpolant::GreedyKernelInterpolant)

Get the history of power function values.

# Returns
- Vector of power function values
"""
get_power_history(interpolant::GreedyKernelInterpolant) = interpolant.power_history

"""
    get_error_history(interpolant::GreedyKernelInterpolant)

Get the history of approximation errors.

# Returns
- Vector of error values
"""
get_error_history(interpolant::GreedyKernelInterpolant) = interpolant.error_history

"""
    get_training_history(interpolant::GreedyKernelInterpolant)

Get the complete training history.

# Returns
- Dictionary containing training history
"""
get_training_history(interpolant::GreedyKernelInterpolant) = interpolant.training_history

"""
    get_n_centers(interpolant::GreedyKernelInterpolant)

Get the number of selected centers.

# Returns
- Number of centers
"""
get_n_centers(interpolant::GreedyKernelInterpolant) = size(interpolant.centers, 1)

"""
    is_fitted(interpolant::GreedyKernelInterpolant)

Check if the interpolant has been fitted.

# Returns
- true if fitted, false otherwise
"""
is_fitted(interpolant::GreedyKernelInterpolant) = !isempty(interpolant.centers)

"""
    continue_fit!(interpolant::GreedyKernelInterpolant, X::AbstractArray, Y::AbstractArray;
                 X_val::AbstractArray=nothing, Y_val::Y_val=nothing, max_iterations::Int=nothing)

Continue fitting the interpolant with additional iterations.

# Arguments
- `interpolant::GreedyKernelInterpolant`: The interpolant to continue fitting
- `X::AbstractArray`: Training input points
- `Y::AbstractArray`: Training target values
- `X_val::AbstractArray`: Validation input points (optional)
- `Y_val::AbstractArray`: Validation target values (optional)
- `max_iterations::Int`: Additional iterations (default: interpolant.max_iterations)

# Returns
- The updated interpolant
"""
function continue_fit!(
        interpolant::GreedyKernelInterpolant, X::AbstractArray, Y::AbstractArray;
        X_val = nothing, Y_val = nothing, max_iterations = nothing
    )

    if !is_fitted(interpolant)
        error("Cannot continue fitting an unfitted interpolant")
    end

    # Store current settings
    original_max_iter = interpolant.max_iterations

    # Set new max iterations
    if max_iterations !== nothing
        interpolant.max_iterations = max_iterations
    end

    # Continue fitting
    result = fit!(interpolant, X, Y; X_val = X_val, Y_val = Y_val)

    # Restore original settings
    interpolant.max_iterations = original_max_iter

    return result
end

"""
    get_sparsity_ratio(interpolant::GreedyKernelInterpolant, X::AbstractArray)

Compute the sparsity ratio of the kernel matrix.

# Arguments
- `interpolant::GreedyKernelInterpolant`: The fitted interpolant
- `X::AbstractArray`: Points to evaluate sparsity on

# Returns
- Sparsity ratio (percentage of zero elements)
"""
function get_sparsity_ratio(interpolant::GreedyKernelInterpolant, X::AbstractArray)
    if !is_fitted(interpolant)
        error("Interpolant must be fitted to compute sparsity")
    end

    # Compute kernel matrix
    K = evalKmatrix(interpolant.kernel, X, X)

    # Count non-zero elements
    total_elements = length(K)
    non_zero_elements = count(!iszero, K)

    # Return sparsity ratio
    return (total_elements - non_zero_elements) / total_elements * 100.0
end

"""
    estimate_convergence_rate(interpolant::GreedyKernelInterpolant)

Estimate the convergence rate from the power function history.

# Returns
- Estimated convergence rate
"""
function estimate_convergence_rate(interpolant::GreedyKernelInterpolant)
    if length(interpolant.power_history) < 10
        return NaN
    end

    # Use last 30% of iterations for rate estimation
    n = length(interpolant.power_history)
    tail_start = max(1, floor(Int, 0.7 * n))

    # Fit exponential decay: p(n) ≈ C * exp(-α * n)
    n_tail = collect(tail_start:n)
    p_tail = interpolant.power_history[tail_start:end]

    # Filter out very small or zero values
    valid_mask = p_tail .> 1.0e-12
    if sum(valid_mask) < 3
        return NaN
    end

    n_valid = n_tail[valid_mask]
    p_valid = p_tail[valid_mask]

    # Linear fit in log space: log(p) ≈ log(C) - α * n
    log_p = log.(p_valid)

    # Linear regression
    A = hcat(ones(length(n_valid)), n_valid)
    coeffs = A \ log_p

    # Extract convergence rate and ensure it's reasonable
    α = -coeffs[2]

    # Check if the fit is reasonable
    if !isfinite(α) || α < 0 || α > 10
        return NaN
    end

    return α
end
