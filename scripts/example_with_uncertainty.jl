using SimpleKernelRegression
using CairoMakie
using Random
using Statistics
using Printf

Random.seed!(42)

println("=== Kernel Regression with Uncertainty Bands ===")

# Generate synthetic data for regression
n_train = 30
n_test = 200
x_range = range(-3, 3, length=n_test)

# Training data with noise - let's use a more interesting function
x_train = rand(n_train) * 6 .- 3
y_true_train = @. sin(x_train) + 0.3 * x_train  # Sinusoidal with linear trend
y_train = y_true_train + 0.3 * randn(n_train)   # Add noise

# Test data
x_test = collect(x_range)
y_true_test = @. sin(x_test) + 0.3 * x_test

# Reshape for kernel regression (needs to be matrices)
X_train = reshape(x_train, :, 1)
Y_train = reshape(y_train, :, 1) 
X_test = reshape(x_test, :, 1)

println("Training points: $n_train")
println("Test points: $n_test")

# Create different kernel regressors with more reasonable parameters
kernels = [
    ("Gaussian", Gaussian(4.0)),
    ("IMQ", Imq(3.0)),
    ("Epanechnikov", Epanechnikov(4.0))
]

# Regularization parameter
reg = 1e-2

# Estimate function norm for error bounds (using cross-validation error as proxy)
# This is a rough estimate - in practice you might know this or estimate it better
f_norm_estimate = 2.0

# Create plots with uncertainty bands
fig = Figure(size = (900, 600))

results = Dict()
colors = [:blue, :green, :purple]

for (i, (name, kernel)) in enumerate(kernels)
    println("\nProcessing $name kernel...")
    
    # Get interpolant
    f_kernel = get_kernel_interpolant(X_train, Y_train, kernel; reg=reg, solver=BackslashSolver)
    
    # Make predictions
    y_pred = f_kernel(X_test)
    
    # Compute power function for uncertainty estimation
    power_vals = power_function(kernel, X_train, X_test, reg)
    
    # Compute error bounds (like GP uncertainty)
    # The power function gives pointwise error bounds: |f(x) - s(x)| ≤ P(x) * ||f||_H_K
    error_bounds_vals = power_vals * f_norm_estimate
    
    # For visualization, we'll use 2σ-style bounds (95% confidence)
    uncertainty_factor = 2.0
    upper_bound = y_pred[:, 1] + uncertainty_factor * error_bounds_vals
    lower_bound = y_pred[:, 1] - uncertainty_factor * error_bounds_vals
    
    # Store results
    results[name] = Dict(
        "predictions" => y_pred[:, 1],
        "power_function" => power_vals,
        "error_bounds" => error_bounds_vals,
        "upper_bound" => upper_bound,
        "lower_bound" => lower_bound,
        "mse" => mean((y_pred[:, 1] - y_true_test).^2)
    )
    
    # Create subplot
    row = (i - 1) ÷ 2 + 1
    col = (i - 1) % 2 + 1
    if i == 3  # Third plot goes in second row, first column
        row, col = 2, 1
    end
    
    ax = Axis(fig[row, col], 
        xlabel = "x", 
        ylabel = "y",
        title = "$name Kernel with Uncertainty Bands"
    )
    
    # Plot uncertainty band first (so it's behind other elements) with more transparency
    band!(ax, x_test, lower_bound, upper_bound, 
          alpha=0.8, color=colors[i], label="95% Uncertainty")
    
    # Plot true function
    lines!(ax, x_test, y_true_test, label="True function", color=:black, linewidth=2)
    
    # Plot training data
    scatter!(ax, x_train, y_train, label="Training data", color=:red, markersize=6, alpha=0.8)
    
    # Plot predictions
    lines!(ax, x_test, y_pred[:, 1], label="$name prediction", color=colors[i], linewidth=2)
    
    axislegend(ax, position=:lt)
    
    # Print statistics
    println("  MSE: $(round(results[name]["mse"], digits=4))")
    println("  Max uncertainty: $(round(maximum(error_bounds_vals), digits=4))")
    println("  Mean uncertainty: $(round(mean(error_bounds_vals), digits=4))")
    println("  Max power function: $(round(maximum(power_vals), digits=4))")
end

# Save the plot
save("kernel_regression_with_uncertainty.png", fig)
println("\n=== Results Summary ===")
println("Plot saved as 'kernel_regression_with_uncertainty.png'")

# Print detailed comparison
println("\nDetailed Comparison:")
println("=" ^ 60)
println(@sprintf("%-15s %-10s %-12s %-12s %-10s", 
                "Kernel", "MSE", "Max Power", "Mean Power", "Max Error"))
println("=" ^ 60)

for (name, result) in results
    println(@sprintf("%-15s %-10.4f %-12.4f %-12.4f %-10.4f",
                    name, result["mse"], 
                    maximum(result["power_function"]), 
                    mean(result["power_function"]),
                    maximum(result["error_bounds"])))
end

println("\nKey Insights:")
println("- Power function P(x) measures interpolation uncertainty")
println("- Large P(x) values indicate high uncertainty (far from training data)")
println("- Small P(x) values indicate low uncertainty (near training data)")
println("- Error bounds: |f(x) - s(x)| ≤ P(x) * ||f||_H_K")
println("- Similar to GP uncertainty, but based on kernel interpolation theory")

display(fig)