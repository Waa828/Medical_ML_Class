import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.spatial.distance import cdist

# ---- Matérn-52 kernel (provided) ----
def matern52_kernel(x1, x2, sigma_f, length_scale):
    """
    计算两组点之间的 Matérn 52 核。
    参数:
    - X1: 第一组点 (N1 x d)。
    - X2: 第二组点 (N2 x d)。
    - length_scale: 控制函数的平滑度。
    - sigma_f: 信号方差（决定函数幅度）。
    返回:
    - 形状为 (N1 x N2) 的核矩阵。
    """
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    dists = cdist(x1, x2, 'euclidean')
    term1 = (1 + np.sqrt(5) * dists / length_scale + (5 * dists ** 2) / (3 * length_scale ** 2))
    matern52 = sigma_f ** 2 * term1 * np.exp(-np.sqrt(5) * dists / length_scale)
    return matern52


# ---- RBF kernel using cdist with 'euclidean' ----
def rbf_kernel(x1, x2, sigma_f=1.0, length_scale=1.0):
    """Compute the RBF kernel between arrays x1 and x2 using the euclidean distance.
    Parameters:
    - x1, x2: 1D arrays of input points. They will be reshaped to 2D if necessary.
    - sigma_f: Signal variance.
    - length_scale: Length scale of the kernel.
    Returns:
    - Kernel matrix computed using the squared Euclidean distances.
    """
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)

    # Compute the Euclidean distances between all pairs of points
    dists = cdist(x1, x2, metric='euclidean')
    # Square the distances because the RBF kernel uses the squared distance
    sqdist = dists ** 2
    return sigma_f**2 * np.exp(-0.5 * sqdist / (length_scale**2))


# ---- Build the GP predictive function ----
def gp_posterior(x_train, y_train, x_test, kernel_func, sigma_f, length_scale, sigma_n):
    """
    x_train, y_train: training data
    x_test: points to predict
    kernel_func: kernel function (rbf_kernel or matern52_kernel)
    sigma_f: signal variance
    length_scale: length-scale
    sigma_n: noise std (note we usually work with sigma_n^2 in the covariance)
    Returns mean_test, std_test for each point in x_test
    """
    # Compute covariance on training data
    K = kernel_func(x_train, x_train, sigma_f, length_scale)
    # Add noise variance on the diagonal
    K += (sigma_n**2) * np.eye(len(x_train))

    # Covariance between training and test
    K_star = kernel_func(x_train, x_test, sigma_f, length_scale)
    # Covariance among test points
    K_star_star = kernel_func(x_test, x_test, sigma_f, length_scale)

    # Invert K
    K_inv = inv(K)

    # Posterior mean for the function values at x_test
    mean_test = K_star.T.dot(K_inv).dot(y_train)

    # Posterior covariance for the function values at x_test
    cov_test = K_star_star - K_star.T.dot(K_inv).dot(K_star)

    # Standard deviation at test points
    std_test = np.sqrt(np.diag(cov_test))

    return mean_test, std_test


# ---- Synthetic data generator ----
def f_true(x):
    """True function (for demo)."""
    return np.sin(x)


# ---- Generate training data with noise ----
np.random.seed(42)
sigma_n = 0.1  # noise level
x_train = np.linspace(-3, 0, 10)
y_train_noisy = f_true(x_train) + sigma_n * np.random.randn(len(x_train))

# Define a finer grid for predictions
x_test = np.linspace(-4, 4, 200)

# Hyperparameters for comparison
sigma_f = 1.0
length_scale = 1.0

# ---- Perform GP regression with both kernels ----
mean_rbf, std_rbf = gp_posterior(x_train, y_train_noisy, x_test,
                                  rbf_kernel, sigma_f, length_scale, sigma_n)

mean_matern, std_matern = gp_posterior(x_train, y_train_noisy, x_test,
                                        matern52_kernel, sigma_f, length_scale, sigma_n)

# ---- Visualization: Compare RBF and Matérn-52 kernels ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: RBF kernel
ax1 = axes[0]
ax1.plot(x_train, y_train_noisy, 'ro', markersize=8, label="Training data (noisy)")
ax1.plot(x_test, f_true(x_test), 'g--', linewidth=2, alpha=0.7, label="True function: sin(x)")
ax1.plot(x_test, mean_rbf, 'b-', linewidth=2, label="GP mean (RBF)")
ax1.fill_between(x_test, mean_rbf - 2*std_rbf, mean_rbf + 2*std_rbf,
                 color='lightblue', alpha=0.5, label="95% confidence interval")
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("y", fontsize=12)
ax1.set_title("RBF Kernel GP Regression\n" + f"(sigma_f={sigma_f}, length_scale={length_scale}, sigma_n={sigma_n})")
ax1.legend(loc='upper right')
ax1.set_ylim(-2, 2)
ax1.grid(True, alpha=0.3)

# Plot 2: Matérn-52 kernel
ax2 = axes[1]
ax2.plot(x_train, y_train_noisy, 'ro', markersize=8, label="Training data (noisy)")
ax2.plot(x_test, f_true(x_test), 'g--', linewidth=2, alpha=0.7, label="True function: sin(x)")
ax2.plot(x_test, mean_matern, 'b-', linewidth=2, label="GP mean (Matérn-52)")
ax2.fill_between(x_test, mean_matern - 2*std_matern, mean_matern + 2*std_matern,
                 color='lightcoral', alpha=0.5, label="95% confidence interval")
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("y", fontsize=12)
ax2.set_title("Matérn-52 Kernel GP Regression\n" + f"(sigma_f={sigma_f}, length_scale={length_scale}, sigma_n={sigma_n})")
ax2.legend(loc='upper right')
ax2.set_ylim(-2, 2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('P2Q2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- Analysis and Comparison ----
print("=" * 60)
print("Comparison between RBF and Matérn-52 Kernels")
print("=" * 60)
print(f"\nHyperparameters:")
print(f"  sigma_f = {sigma_f}")
print(f"  length_scale = {length_scale}")
print(f"  sigma_n (noise std) = {sigma_n}")
print(f"\nTraining data: y = sin(x) + noise, x in [-3, 0]")
print(f"Test data: x in [-4, 4]")

# Calculate prediction errors on test points (excluding extrapolation region)
test_mask = (x_test >= -3) & (x_test <= 0)  # Interpolation region
y_true_test = f_true(x_test)

mse_rbf_interp = np.mean((mean_rbf[test_mask] - y_true_test[test_mask])**2)
mse_matern_interp = np.mean((mean_matern[test_mask] - y_true_test[test_mask])**2)

print(f"\nMean Squared Error (Interpolation region [-3, 0]):")
print(f"  RBF:       {mse_rbf_interp:.6f}")
print(f"  Matérn-52: {mse_matern_interp:.6f}")

# Extrapolation errors
extrap_mask_left = x_test < -3
extrap_mask_right = x_test > 0

mse_rbf_extrap_left = np.mean((mean_rbf[extrap_mask_left] - y_true_test[extrap_mask_left])**2) if np.any(extrap_mask_left) else 0
mse_matern_extrap_left = np.mean((mean_matern[extrap_mask_left] - y_true_test[extrap_mask_left])**2) if np.any(extrap_mask_left) else 0

mse_rbf_extrap_right = np.mean((mean_rbf[extrap_mask_right] - y_true_test[extrap_mask_right])**2) if np.any(extrap_mask_right) else 0
mse_matern_extrap_right = np.mean((mean_matern[extrap_mask_right] - y_true_test[extrap_mask_right])**2) if np.any(extrap_mask_right) else 0

print(f"\nMean Squared Error (Extrapolation region [0, 4]):")
print(f"  RBF:       {mse_rbf_extrap_right:.6f}")
print(f"  Matérn-52: {mse_matern_extrap_right:.6f}")

# Average uncertainty (standard deviation)
print(f"\nAverage Uncertainty (std) in Extrapolation region [0, 4]:")
print(f"  RBF:       {np.mean(std_rbf[extrap_mask_right]):.4f}")
print(f"  Matérn-52: {np.mean(std_matern[extrap_mask_right]):.4f}")
