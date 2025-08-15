#!/usr/bin/env julia

"""
Power Function Error Bounds Example

This script demonstrates the power function error bounds and error analysis
features for kernel regression, based on the theory from:
"Approximation with Kernel Methods" by Dr. Gabriele Santin & Prof. Haasdonk

The power function P(x) provides error bounds for kernel interpolation:
|f(x) - s(x)| ≤ P(x) * ||f||_H_K

where s(x) is the kernel interpolant and ||f||_H_K is the native space norm.
"""

using SimpleKernelRegression
using CairoMakie
using Random
using LinearAlgebra
using Statistics
using Printf

# Set random seed for reproducibility
Random.seed!(42)

println("=== Power Function Error Bounds Example ===\n")

# Generate test data
println("1. Generating test data...")
n_train = 50
n_test = 200
d = 2

# Training points on a grid
x_range = LinRange(-2.0, 2.0,ceil(Int64,sqrt(n_train))) |> collect 
y_range = LinRange(-2.0, 2.0,ceil(Int64,sqrt(n_train))) |> collect
X_train = hcat([x for x in x_range for y in y_range], 
                [y for x in x_range for y in y_range])

# Test points (randomly distributed)
X_test = 4.0 * rand(n_test, d) .- 2.0

f_true(x) = sin(π * x[1]) + x[2] 
Y_train = [f_true(X_train[i, :]) for i in 1:size(X_train, 1)]
Y_train = reshape(Y_train, :, 1)  # Make it a column vector


println("   Training points: $(size(X_train, 1))")
println("   Test points: $(size(X_test, 1))")
println("   Input dimension: $d")

# Test different kernels
kernels = [
    ("Gaussian", Gaussian(5.0)),
    ("IMQ", Imq(5.0)),
    ("Multiquadratic", Mq(0.1)),
    ("Epanechnikov", Epanechnikov(8.0))
]

println("\n2. Computing power functions and error bounds...")

# Domain for fill distance computation (uniform grid covering test area)
x_domain = range(-2.5, 2.5, length=50)
y_domain = range(-2.5, 2.5, length=50)
Ω = hcat([x for x in x_domain for y in y_domain], 
          [y for x in x_domain for y in y_domain])

results = Dict{String, Dict{String, Any}}()

for (name, kernel) in kernels
    println("   Computing for $name kernel...")
    
    # Get interpolant
    interpolant = get_kernel_interpolant(X_train, Y_train, kernel; reg=1e-6, solver=BackslashSolver)
    
    # Predictions
    Y_pred = interpolant(X_test)
    
    # Power function
    power_vals = power_function(kernel, X_train, X_test, 1e-6)
    
    # Error bounds (assuming ||f||_H_K = 1 for simplicity)
    f_norm = 1.0
    error_bounds_vals = error_bounds(kernel, X_train, X_test, f_norm; reg=1e-6)
    
    # Geometric measures
    h = fill_distance(X_train, Ω)
    q = separation_distance(X_train)
    ρ = mesh_ratio(X_train, Ω)
    
    # Theoretical error bounds
    theoretical_bound, _ = theoretical_error_bounds(kernel, X_train, X_test, f_norm, Ω)
    
    # Actual errors
    actual_errors = [abs(f_true(X_test[i, :]) - Y_pred[i, 1]) for i in 1:size(X_test, 1)]
    
    # Cross-validation error
    cv_error = cross_validation_error(kernel, X_train, Y_train; k_folds=5, reg=1e-6)
    
    results[name] = Dict(
        "power_vals" => power_vals,
        "error_bounds" => error_bounds_vals,
        "theoretical_bound" => theoretical_bound,
        "actual_errors" => actual_errors,
        "fill_distance" => h,
        "separation_distance" => q,
        "mesh_ratio" => ρ,
        "cv_error" => cv_error,
        "max_power" => maximum(power_vals),
        "mean_power" => mean(power_vals),
        "max_actual_error" => maximum(actual_errors),
        "mean_actual_error" => mean(actual_errors)
    )
    
    println("     Fill distance: $(round(h, digits=4))")
    println("     Separation distance: $(round(q, digits=4))")
    println("     Mesh ratio: $(round(ρ, digits=4))")
    println("     Max power function: $(round(maximum(power_vals), digits=4))")
    println("     CV error: $(round(cv_error, digits=6))")
end

# Analysis and visualization
println("\n3. Analysis and visualization...")

# Create Makie figure with subplots
fig = Figure(size=(1200, 800))

# Plot 1: Power function values
ax1 = Axis(fig[1, 1], 
    title="Power Function Values by Kernel",
    xlabel="Test Point Index", 
    ylabel="Power Function P(x)"
)

# Plot 2: Error bounds vs actual errors  
ax2 = Axis(fig[1, 2],
    title="Error Bounds vs Actual Errors",
    xlabel="Test Point Index", 
    ylabel="Error"
)

# Plot 3: Scatter plot
ax3 = Axis(fig[2, 1],
    title="Power Function vs Actual Error",
    xlabel="Power Function P(x)", 
    ylabel="Actual Error |f(x) - s(x)|"
)

# Plot 4: Summary statistics
ax4 = Axis(fig[2, 2],
    title="Kernel Performance Comparison",
    xlabel="Kernel Type", 
    ylabel="Value"
)

colors = [:blue, :red, :green, :orange]
kernel_names = String[]
max_powers = Float64[]
mean_powers = Float64[]
cv_errors = Float64[]

for (i, (name, result)) in enumerate(results)
    color = colors[i]
    push!(kernel_names, name)
    push!(max_powers, result["max_power"])
    push!(mean_powers, result["mean_power"])
    push!(cv_errors, result["cv_error"])
    
    # Plot 1: Power function values
    lines!(ax1, 1:length(result["power_vals"]), result["power_vals"], 
           color=color, linewidth=2, label=name)
    
    # Plot 2: Error bounds vs actual errors
    lines!(ax2, 1:length(result["error_bounds"]), result["error_bounds"], 
           color=color, linewidth=2, linestyle=:dash, label="$(name) Bounds")
    lines!(ax2, 1:length(result["actual_errors"]), result["actual_errors"], 
           color=color, linewidth=2, label="$(name) Actual")
    
    # Plot 3: Scatter plot
    scatter!(ax3, result["power_vals"], result["actual_errors"], 
             color=color, markersize=8, alpha=0.6, label=name)
end

# Add theoretical bounds line to scatter plot
max_power = maximum([result["max_power"] for result in values(results)])
max_error = maximum([result["max_actual_error"] for result in values(results)])
lines!(ax3, [0, max_power], [0, max_error], 
       color=:black, linestyle=:dash, linewidth=3, label="Theoretical Bound")

# Plot 4: Bar chart comparison
barplot!(ax4, 1:length(kernel_names), max_powers, 
         color=colors, label="Max Power", alpha=0.7)
barplot!(ax4, 1:length(kernel_names), mean_powers, 
         color=colors, label="Mean Power", alpha=0.5, offset=0.2)

# Set x-axis labels for bar chart
ax4.xticks = (1:length(kernel_names), kernel_names)

# Add legends
axislegend(ax1, position=:rt)
axislegend(ax2, position=:rt)
axislegend(ax3, position=:rt)
axislegend(ax4, position=:rt)

# Summary table
println("\n4. Summary of Results:")
println("=" ^ 80)
println(@sprintf("%-15s %-12s %-12s %-12s %-12s %-12s", 
                "Kernel", "Max Power", "Mean Power", "Max Error", "Mean Error", "CV Error"))
println("=" ^ 80)

for (name, result) in results
    println(@sprintf("%-15s %-12.4f %-12.4f %-12.4f %-12.4f %-12.6f",
                    name, result["max_power"], result["mean_power"], 
                    result["max_actual_error"], result["mean_actual_error"], 
                    result["cv_error"]))
end

println("\n5. Geometric Measures:")
println("=" ^ 50)
for (name, result) in results
    println(@sprintf("%-15s: h=%.4f, q=%.4f, ρ=%.4f", 
                    name, result["fill_distance"], result["separation_distance"], 
                    result["mesh_ratio"]))
end

# Theoretical analysis
println("\n6. Theoretical Error Bounds:")
println("=" ^ 40)
for (name, result) in results
    println(@sprintf("%-15s: Theoretical bound = %.6f", name, result["theoretical_bound"]))
end

# Validation of error bounds
println("\n7. Error Bound Validation:")
println("=" ^ 30)
for (name, result) in results
    violations = sum(result["actual_errors"] .> result["error_bounds"])
    total = length(result["actual_errors"])
    violation_rate = violations / total * 100
    
    println(@sprintf("%-15s: %d/%d violations (%.1f%%)", 
                    name, violations, total, violation_rate))
end

# Save plot
println("\n8. Saving plot...")
save("power_function_analysis.png", fig)
println("   Plot saved as: power_function_analysis.png")

# Display plot
display(fig)

