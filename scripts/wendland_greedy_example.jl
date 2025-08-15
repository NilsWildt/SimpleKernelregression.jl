#!/usr/bin/env julia

"""
Wendland Kernels and Greedy Interpolation Example

This script demonstrates the new Wendland kernels and greedy interpolation features,
based on the VKOGA implementation from the University of Stuttgart.

Features demonstrated:
1. Wendland kernels with automatic bandwidth selection
2. Compact support and sparsity
3. Greedy interpolation with power function selection
4. Comparison with standard kernel interpolation
5. Visualization of convergence and sparsity
"""

using SimpleKernelRegression
using CairoMakie
using Random
using LinearAlgebra
using Statistics
using Printf

# Set random seed for reproducibility
Random.seed!(42)

println("=== Wendland Kernels and Greedy Interpolation Example ===\n")

# Generate test data
println("1. Generating test data...")
n_train = 500
n_test = 1000
d = 2

# Training points on a grid with some noise
x_range = range(-3.0, 3.0, length=ceil(Int64,sqrt(n_train)))
y_range = range(-3.0, 3.0, length=ceil(Int64,sqrt(n_train)))
X_train = hcat([x for x in x_range for y in y_range], 
                [y for x in x_range for y in y_range])

# Add some noise to break the grid structure
X_train .+= 0.1 * randn(size(X_train))

# Test points (randomly distributed)
X_test = 6.0 * rand(n_test, d) .- 3.0

# True function: f(x,y) = sin(πx) * cos(πy) + 0.1 * noise
f_true(x) = sin(π * x[1]) + x[2]
Y_train = [f_true(X_train[i, :]) for i in 1:size(X_train, 1)]
Y_train = reshape(Y_train, :, 1)  # Make it a column vector

# Add noise to training targets
Y_train .+= 0.05 * randn(size(Y_train))

println("   Training points: $(size(X_train, 1))")
println("   Test points: $(size(X_test, 1))")
println("   Input dimension: $d")

# Test different kernels including Wendland
println("\n2. Testing different kernels...")

kernels = [
    ("Gaussian", Gaussian(0.5)),
    ("IMQ", Imq(1.0)),
    ("Wendland k=0", Wendland(1.0, 0, d)),
    ("Wendland k=1", Wendland(1.0, 1, d)),
    ("Wendland k=2", Wendland(1.0, 2, d)),
    ("Wendland k=3", Wendland(1.0, 3, d))
]

# Test automatic bandwidth selection for Wendland
println("\n3. Testing automatic bandwidth selection...")
wendland_auto = Wendland(X_train, 2, d; coverage_factor=2.5)
println("   Auto-selected bandwidth: $(round(get_support_radius(wendland_auto), digits=4))")
println("   Smoothness parameter: $(get_smoothness(wendland_auto))")
println("   Dimension: $(get_dimension(wendland_auto))")

# Test standard interpolation
println("\n4. Testing standard kernel interpolation...")
results_standard = Dict{String, Dict{String, Any}}()

for (name, kernel) in kernels
    println("   Testing $name kernel...")
    
    # Get interpolant
    interpolant = get_kernel_interpolant(X_train, Y_train, kernel; reg=1e-6)
    
    # Predictions
    Y_pred = interpolant(X_test)
    
    # Compute errors
    actual_errors = [abs(f_true(X_test[i, :]) - Y_pred[i, 1]) for i in 1:size(X_test, 1)]
    
    # Power function analysis
    power_vals = power_function(kernel, X_train, X_test, 1e-6)
    
    # Geometric measures
    domain_points = 6.0 * rand(200, d) .- 3.0  # Random domain points
    h = fill_distance(X_train, domain_points)
    q = separation_distance(X_train)
    ρ = mesh_ratio(X_train, domain_points)
    
    results_standard[name] = Dict(
        "kernel" => kernel,
        "power_vals" => power_vals,
        "actual_errors" => actual_errors,
        "fill_distance" => h,
        "separation_distance" => q,
        "mesh_ratio" => ρ,
        "max_power" => maximum(power_vals),
        "mean_power" => mean(power_vals),
        "max_actual_error" => maximum(actual_errors),
        "mean_actual_error" => mean(actual_errors)
    )
    
    println("     Max power: $(round(maximum(power_vals), digits=4))")
    println("     Mean error: $(round(mean(actual_errors), digits=4))")
    println("     Fill distance: $(round(h, digits=4))")
end

# Test greedy interpolation
println("\n5. Testing greedy interpolation...")
greedy_results = Dict{String, Dict{String, Any}}()

greedy_types = [:p_greedy, :f_greedy, :fp_greedy]
greedy_names = ["Power Greedy", "Function Greedy", "Combined Greedy"]

for (greedy_type, greedy_name) in zip(greedy_types, greedy_names)
    println("   Testing $greedy_name...")
    
    # Create greedy interpolant
    greedy_interp = GreedyKernelInterpolant(
        wendland_auto; 
        tolerance=1e-8, 
        max_iterations=100,
        greedy_type=greedy_type,
        verbose=false
    )
    
    # Fit the interpolant
    fit!(greedy_interp, X_train, Y_train)
    
    # Get results
    n_centers = get_n_centers(greedy_interp)
    power_history = get_power_history(greedy_interp)
    error_history = get_error_history(greedy_interp)
    
    # Predictions
    Y_pred_greedy = predict(greedy_interp, X_test)
    
    # Compute errors
    actual_errors_greedy = [abs(f_true(X_test[i, :]) - Y_pred_greedy[i, 1]) for i in 1:size(X_test, 1)]
    
    # Sparsity analysis
    sparsity_ratio = get_sparsity_ratio(greedy_interp, X_test)
    
    # Convergence rate
    conv_rate = estimate_convergence_rate(greedy_interp)
    
    greedy_results[greedy_name] = Dict(
        "interpolant" => greedy_interp,
        "n_centers" => n_centers,
        "power_history" => power_history,
        "error_history" => error_history,
        "actual_errors" => actual_errors_greedy,
        "sparsity_ratio" => sparsity_ratio,
        "convergence_rate" => conv_rate,
        "max_actual_error" => maximum(actual_errors_greedy),
        "mean_actual_error" => mean(actual_errors_greedy)
    )
    
    println("     Centers selected: $n_centers")
    println("     Final power: $(round(power_history[end], digits=6))")
    println("     Final error: $(round(error_history[end], digits=6))")
    println("     Sparsity: $(round(sparsity_ratio, digits=1))%")
    println("     Convergence rate: $(round(conv_rate, digits=3))")
end

# Visualization
println("\n6. Creating visualizations...")

# Create Makie figure with subplots
fig = Figure(size=(1400, 1000))

# Plot 1: Kernel comparison
ax1 = Axis(fig[1, 1], 
    title="Kernel Performance Comparison",
    xlabel="Kernel Type", 
    ylabel="Value"
)

# Prepare data for bar plot
kernel_names = String[]
max_powers = Float64[]
mean_errors = Float64[]

for (name, result) in results_standard
    push!(kernel_names, name)
    push!(max_powers, result["max_power"])
    push!(mean_errors, result["mean_actual_error"])
end

# Bar plot - filter out NaN values
valid_max_powers = replace(max_powers, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
valid_mean_errors = replace(mean_errors, NaN => 0.0, Inf => 0.0, -Inf => 0.0)

barplot!(ax1, 1:length(kernel_names), valid_max_powers, 
         color=:blue, label="Max Power", alpha=0.7)
barplot!(ax1, 1:length(kernel_names), valid_mean_errors, 
         color=:red, label="Mean Error", alpha=0.7, offset=0.2)

ax1.xticks = (1:length(kernel_names), kernel_names)

# Plot 2: Greedy convergence
ax2 = Axis(fig[1, 2],
    title="Greedy Interpolation Convergence",
    xlabel="Iteration", 
    ylabel="Value",
    yscale=log10
)

colors = [:blue, :red, :green]
for (i, (name, result)) in enumerate(greedy_results)
    color = colors[i]
    iterations = 1:length(result["power_history"])
    
    # Filter out NaN and infinite values from history
    power_history = result["power_history"]
    error_history = result["error_history"]
    
    power_valid = isfinite.(power_history)
    error_valid = isfinite.(error_history)
    
    if any(power_valid)
        lines!(ax2, iterations[power_valid], power_history[power_valid], 
               color=color, linewidth=2, label="$(name) Power")
    end
    if any(error_valid)
        lines!(ax2, iterations[error_valid], error_history[error_valid], 
               color=color, linewidth=2, linestyle=:dash, label="$(name) Error")
    end
end

# Plot 3: Power function vs actual error
ax3 = Axis(fig[2, 1],
    title="Power Function vs Actual Error",
    xlabel="Power Function P(x)", 
    ylabel="Actual Error |f(x) - s(x)|"
)

# Scatter plot for standard kernels
for (name, result) in results_standard
    # Filter out NaN and infinite values
    power_vals = result["power_vals"]
    actual_errors = result["actual_errors"]
    valid_mask = isfinite.(power_vals) .& isfinite.(actual_errors)
    if any(valid_mask)
        scatter!(ax3, power_vals[valid_mask], actual_errors[valid_mask], 
                 color=:gray, markersize=4, alpha=0.4, label="")
    end
end

# Scatter plot for greedy results
for (i, (name, result)) in enumerate(greedy_results)
    color = colors[i]
    # Use final power function values
    final_power = power_function(wendland_auto, get_centers(result["interpolant"]), X_test, 1e-6)
    actual_errors = result["actual_errors"]
    # Filter out NaN and infinite values
    valid_mask = isfinite.(final_power) .& isfinite.(actual_errors)
    if any(valid_mask)
        scatter!(ax3, final_power[valid_mask], actual_errors[valid_mask], 
                 color=color, markersize=6, alpha=0.7, label=name)
    end
end

# Plot 4: Sparsity analysis
ax4 = Axis(fig[2, 2],
    title="Sparsity and Efficiency Analysis",
    xlabel="Greedy Type", 
    ylabel="Value"
)

greedy_names_short = ["Power", "Function", "Combined"]
n_centers_data = [result["n_centers"] for result in values(greedy_results)]
sparsity_data = [result["sparsity_ratio"] for result in values(greedy_results)]

# Filter out any potential NaN values
valid_n_centers = replace(n_centers_data, NaN => 0, Inf => 0, -Inf => 0)
valid_sparsity = replace(sparsity_data, NaN => 0.0, Inf => 0.0, -Inf => 0.0)

barplot!(ax4, 1:length(greedy_names_short), valid_n_centers, 
         color=:blue, label="Number of Centers", alpha=0.7)
barplot!(ax4, 1:length(greedy_names_short), valid_sparsity, 
         color=:green, label="Sparsity %", alpha=0.7, offset=0.2)

ax4.xticks = (1:length(greedy_names_short), greedy_names_short)

# Plot 5: Center selection visualization
ax5 = Axis(fig[3, 1],
    title="Selected Centers by Greedy Algorithm",
    xlabel="X", 
    ylabel="Y"
)

# Plot training points
scatter!(ax5, X_train[:, 1], X_train[:, 2], 
         color=:lightgray, markersize=2, alpha=0.3, label="Training Points")

# Plot selected centers for each greedy type
for (i, (name, result)) in enumerate(greedy_results)
    color = colors[i]
    centers = get_centers(result["interpolant"])
    
    # Filter out any NaN or infinite coordinates
    if !isempty(centers) && size(centers, 2) >= 2
        x_coords = centers[:, 1]
        y_coords = centers[:, 2]
        valid_mask = isfinite.(x_coords) .& isfinite.(y_coords)
        
        if any(valid_mask)
            scatter!(ax5, x_coords[valid_mask], y_coords[valid_mask], 
                     color=color, markersize=8, alpha=0.8, label=name)
        end
    end
end

# Plot 6: Convergence rate comparison
ax6 = Axis(fig[3, 2],
    title="Convergence Rate Analysis",
    xlabel="Greedy Type", 
    ylabel="Convergence Rate α"
)

conv_rates = [result["convergence_rate"] for result in values(greedy_results)]
# Filter out NaN values for plotting
valid_rates = filter(!isnan, conv_rates)
valid_names = greedy_names_short[.!isnan.(conv_rates)]

if !isempty(valid_rates)
    barplot!(ax6, 1:length(valid_names), valid_rates, 
             color=:purple, alpha=0.7)
    ax6.xticks = (1:length(valid_names), valid_names)
else
    # If no valid rates, just show empty plot
    ax6.title = "Convergence Rate Analysis (No valid rates)"
end

# Skip legends temporarily to test the rest of the script
# try
#     axislegend(ax1, position=:rt)
# catch e
#     println("Warning: Could not create legend for ax1: $e")
# end
# 
# try
#     axislegend(ax2, position=:rt)
# catch e
#     println("Warning: Could not create legend for ax2: $e")
# end
# 
# try
#     axislegend(ax3, position=:rt)
# catch e
#     println("Warning: Could not create legend for ax3: $e")
# end
# 
# try
#     axislegend(ax4, position=:rt)
# catch e
#     println("Warning: Could not create legend for ax4: $e")
# end
# 
# try
#     axislegend(ax5, position=:rt)
# catch e
#     println("Warning: Could not create legend for ax5: $e")
# end

# Summary table
println("\n7. Summary of Results:")
println("=" ^ 100)
println(@sprintf("%-20s %-15s %-15s %-15s %-15s %-15s", 
                "Kernel", "Max Power", "Mean Error", "Fill Dist", "Sep Dist", "Mesh Ratio"))
println("=" ^ 100)

for (name, result) in results_standard
    println(@sprintf("%-20s %-15.4f %-15.4f %-15.4f %-15.4f %-15.4f",
                    name, result["max_power"], result["mean_actual_error"], 
                    result["fill_distance"], result["separation_distance"], 
                    result["mesh_ratio"]))
end

println("\nGreedy Interpolation Results:")
println("=" ^ 60)
println(@sprintf("%-20s %-15s %-15s %-15s %-15s", 
                "Greedy Type", "Centers", "Final Power", "Final Error", "Sparsity %"))
println("=" ^ 60)

for (name, result) in greedy_results
    println(@sprintf("%-20s %-15d %-15.6f %-15.6f %-15.1f",
                    name, result["n_centers"], 
                    result["power_history"][end], 
                    result["error_history"][end],
                    result["sparsity_ratio"]))
end

# Save plot
println("\n8. Saving plot...")
save("wendland_greedy_analysis.png", fig)
println("   Plot saved as: wendland_greedy_analysis.png")

# Display plot
display(fig)

println("\n=== Example completed successfully! ===")
println("\nKey insights:")
println("- Wendland kernels provide compact support and automatic bandwidth selection")
println("- Greedy interpolation selects the most informative points using power function")
println("- Power greedy focuses on geometric coverage, function greedy on approximation error")
println("- Combined greedy balances both objectives")
println("- Compact support leads to sparse matrices and computational efficiency")
