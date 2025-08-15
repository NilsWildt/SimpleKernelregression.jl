using SimpleKernelRegression
using CairoMakie
using Random
using Statistics

Random.seed!(42)

# Generate synthetic 3D data for regression
n_train = 100
n_test = 50
x_range = range(-2, 2, length=n_test)
y_range = range(-2, 2, length=n_test)

# Training data with noise
x1_train = rand(n_train) * 4 .- 2  # x coordinates
x2_train = rand(n_train) * 4 .- 2  # y coordinates
y_true_train = 0.5 * (x1_train.^2 + x2_train.^2) .+ 0.3 * x1_train .* x2_train .+ 2
y_train = y_true_train + 0.3 * randn(n_train)

# Test data grid
x1_test = repeat(x_range, outer=n_test)
x2_test = repeat(y_range, inner=n_test)
y_true_test = 0.5 * (x1_test.^2 + x2_test.^2) .+ 0.3 * x1_test .* x2_test .+ 2

# Reshape for kernel regression (needs to be matrices)
X_train = hcat(x1_train, x2_train)  # (n_train, 2)
Y_train = reshape(y_train, :, 1)    # (n_train, 1)
X_test = hcat(x1_test, x2_test)     # (n_test^2, 2)

# Create different kernel regressors
gaussian_kernel = Gaussian(3.0)
linear_kernel = Linear(true)  
imq_kernel = Imq(5.0)
epanechnikov_kernel = Epanechnikov(8.0)

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

# Reshape predictions for 3D plotting
y_pred_gaussian_3d = reshape(y_pred_gaussian[:, 1], n_test, n_test)
y_pred_linear_3d = reshape(y_pred_linear[:, 1], n_test, n_test)
y_pred_imq_3d = reshape(y_pred_imq[:, 1], n_test, n_test)
y_pred_epanechnikov_3d = reshape(y_pred_epanechnikov[:, 1], n_test, n_test)
y_true_test_3d = reshape(y_true_test, n_test, n_test)

# Create 3D plot
fig = Figure(size = (1200, 800))

# Plot 1: True function
ax1 = Axis3(fig[1, 1], 
    xlabel = "x₁", 
    ylabel = "x₂",
    zlabel = "y",
    title = "True Function: 0.5(x₁² + x₂²) + 0.3x₁x₂ + 2 (Cup Shape)"
)
surface!(ax1, x_range, y_range, y_true_test_3d, colormap=:viridis, alpha=0.8)

# Plot 2: Training data
ax2 = Axis3(fig[1, 2], 
    xlabel = "x₁", 
    ylabel = "x₂",
    zlabel = "y",
    title = "Training Data with Noise"
)
scatter!(ax2, x1_train, x2_train, y_train, color=:red, markersize=8, label="Training points")

# Plot 3: Gaussian kernel prediction
ax3 = Axis3(fig[2, 1], 
    xlabel = "x₁", 
    ylabel = "x₂",
    zlabel = "y",
    title = "Gaussian Kernel Prediction"
)
surface!(ax3, x_range, y_range, y_pred_gaussian_3d, colormap=:plasma, alpha=0.8)

# Plot 4: Linear kernel prediction
ax4 = Axis3(fig[2, 2], 
    xlabel = "x₁", 
    ylabel = "x₂",
    zlabel = "y",
    title = "Linear Kernel Prediction"
)
surface!(ax4, x_range, y_range, y_pred_linear_3d, colormap=:inferno, alpha=0.8)

# Plot 5: IMQ kernel prediction
ax5 = Axis3(fig[3, 1], 
    xlabel = "x₁", 
    ylabel = "x₂",
    zlabel = "y",
    title = "IMQ Kernel Prediction"
)
surface!(ax5, x_range, y_range, y_pred_imq_3d, colormap=:magma, alpha=0.8)

# Plot 6: Epanechnikov kernel prediction
ax6 = Axis3(fig[3, 2], 
    xlabel = "x₁", 
    ylabel = "x₂",
    zlabel = "y",
    title = "Epanechnikov Kernel Prediction"
)
surface!(ax6, x_range, y_range, y_pred_epanechnikov_3d, colormap=:turbo, alpha=0.8)

# Save the plot
save("kernel_regression_3d_example.png", fig, px_per_unit=2)
println("3D plot saved as 'kernel_regression_3d_example.png'")

# Print some statistics
println("\n3D Kernel Regression Results:")
println(repeat("=", 50))
println("Mean Squared Error (Gaussian): ", mean((y_pred_gaussian[:, 1] - y_true_test).^2))
println("Mean Squared Error (Linear): ", mean((y_pred_linear[:, 1] - y_true_test).^2))
println("Mean Squared Error (IMQ): ", mean((y_pred_imq[:, 1] - y_true_test).^2))
println("Mean Squared Error (Epanechnikov): ", mean((y_pred_epanechnikov[:, 1] - y_true_test).^2))

# Additional analysis
println("\nKernel Performance Analysis:")
println(repeat("=", 50))
println("Training points: ", n_train)
println("Test grid size: ", n_test, " × ", n_test, " = ", n_test^2, " points")
println("Data range: x₁ ∈ [-2, 2], x₂ ∈ [-2, 2]")
println("True function: y = 0.5(x₁² + x₂²) + 0.3x₁x₂ + 2 (Cup Shape)")
println("Noise level: σ = 0.3")
