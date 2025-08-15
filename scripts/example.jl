using SimpleKernelRegression
using CairoMakie
using Random
using Statistics

Random.seed!(42)

# Generate synthetic data for linear regression
n_train = 50
n_test = 500
x_range = range(-3, 3, length=n_test)

# Training data with noise
x_train = rand(n_train) * 6 .- 3
y_true_train = 2 * x_train .+ 1
y_train = y_true_train + 0.5 * randn(n_train)

# Test data
x_test = collect(x_range)
y_true_test = 2 * x_test .+ 1

# Reshape for kernel regression (needs to be matrices)
X_train = reshape(x_train, :, 1)
Y_train = reshape(y_train, :, 1) 
X_test = reshape(x_test, :, 1)

# Create different kernel regressors
gaussian_kernel = Gaussian(1.0)
linear_kernel = Linear(true)  
imq_kernel = Imq(1.0)
epanechnikov_kernel = Epanechnikov(5.0)  # Bandwidth parameter - controls the support region

# Get interpolants
reg = 1e-2
f_gaussian = get_kernel_interpolant(X_train, Y_train, gaussian_kernel; reg=reg, solver=BackslashSolver)
f_linear = get_kernel_interpolant(X_train, Y_train, linear_kernel; reg=reg, solver=BackslashSolver)
f_imq = get_kernel_interpolant(X_train, Y_train, imq_kernel; reg=reg, solver=BackslashSolver)
f_epanechnikov = get_kernel_interpolant(X_train, Y_train, epanechnikov_kernel; reg=reg, solver=BackslashSolver)

# Make predictions
y_pred_gaussian = f_gaussian(X_test)
y_pred_linear = f_linear(X_test)
y_pred_imq = f_imq(X_test)
y_pred_epanechnikov = f_epanechnikov(X_test)

# Create plot
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], 
    xlabel = "x", 
    ylabel = "y",
    title = "Kernel Regression Comparison"
)

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