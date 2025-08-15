using SimpleKernelRegression
using CairoMakie
using Random
using Statistics

Random.seed!(42)

# Generate synthetic data for regression
n_train = 50
n_test = 500
x_range = range(-3, 3, length=n_test)

# Training data with noise - using a more interesting function
x_train = rand(n_train) * 6 .- 3
y_true_train = @. sin(x_train) + 0.3 * x_train  # Sinusoidal with linear trend
y_train = y_true_train + 0.3 * randn(n_train)

# Test data
x_test = collect(x_range)
y_true_test = @. sin(x_test) + 0.3 * x_test

# Reshape for kernel regression (needs to be matrices)
X_train = reshape(x_train, :, 1)
Y_train = reshape(y_train, :, 1) 
X_test = reshape(x_test, :, 1)

# Create different kernel regressors
gaussian_kernel = Gaussian(0.8)
linear_kernel = Linear(true)  
imq_kernel = Imq(1.0)
epanechnikov_kernel = Epanechnikov(2.0)

# Get interpolants
reg = 1e-3
f_gaussian = get_kernel_interpolant(X_train, Y_train, gaussian_kernel; reg=reg, solver=BackslashSolver)
f_linear = get_kernel_interpolant(X_train, Y_train, linear_kernel; reg=reg, solver=BackslashSolver)
f_imq = get_kernel_interpolant(X_train, Y_train, imq_kernel; reg=reg, solver=BackslashSolver)
f_epanechnikov = get_kernel_interpolant(X_train, Y_train, epanechnikov_kernel; reg=reg, solver=BackslashSolver)

# Make predictions
y_pred_gaussian = f_gaussian(X_test)
y_pred_linear = f_linear(X_test)
y_pred_imq = f_imq(X_test)
y_pred_epanechnikov = f_epanechnikov(X_test)

# Add uncertainty bands for Gaussian kernel (like GP)
power_vals_gaussian = power_function(gaussian_kernel, X_train, X_test, reg)
f_norm_estimate = 1.5  # Estimated function norm
uncertainty_gaussian = power_vals_gaussian * f_norm_estimate * 2.0  # 2σ-style bounds
upper_bound_gaussian = y_pred_gaussian[:, 1] + uncertainty_gaussian
lower_bound_gaussian = y_pred_gaussian[:, 1] - uncertainty_gaussian

# Create plot
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], 
    xlabel = "x", 
    ylabel = "y",
    title = "Kernel Regression with Uncertainty (GP-style)"
)

# Plot uncertainty band first (behind other elements)
band!(ax, x_test, lower_bound_gaussian, upper_bound_gaussian, 
      alpha=0.15, color=:lightblue, label="Gaussian 95% uncertainty")

# Plot true function
lines!(ax, x_test, y_true_test, label="True function", color=:black, linewidth=3)

# Plot training data
scatter!(ax, x_train, y_train, label="Training data", color=:red, markersize=8)

# Plot predictions
lines!(ax, x_test, y_pred_gaussian[:, 1], label="Gaussian kernel", color=:blue, linewidth=2)
lines!(ax, x_test, y_pred_linear[:, 1], label="Linear kernel", color=:green, linewidth=2)
lines!(ax, x_test, y_pred_imq[:, 1], label="IMQ kernel", color=:orange, linewidth=2)
lines!(ax, x_test, y_pred_epanechnikov[:, 1], label="Epanechnikov kernel", color=:purple, linewidth=2)

axislegend(ax, position=:lt)

# Save the plot
save("kernel_regression_example.png", fig)
println("Plot saved as 'kernel_regression_example.png'")

# Print some statistics
println("Mean Squared Error (Gaussian): ", mean((y_pred_gaussian[:, 1] - y_true_test).^2))
println("Mean Squared Error (Linear): ", mean((y_pred_linear[:, 1] - y_true_test).^2))
println("Mean Squared Error (IMQ): ", mean((y_pred_imq[:, 1] - y_true_test).^2))
println("Mean Squared Error (Epanechnikov): ", mean((y_pred_epanechnikov[:, 1] - y_true_test).^2))

# Print uncertainty statistics
println("\nUncertainty Analysis (Gaussian kernel):")
println("Max uncertainty: ", round(maximum(uncertainty_gaussian), digits=4))
println("Mean uncertainty: ", round(mean(uncertainty_gaussian), digits=4))
println("Power function range: [", round(minimum(power_vals_gaussian), digits=4), ", ", round(maximum(power_vals_gaussian), digits=4), "]")