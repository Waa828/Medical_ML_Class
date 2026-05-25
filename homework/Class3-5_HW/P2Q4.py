"""
Bayesian Optimization with Expected Improvement (EI) Acquisition Function

This script implements Bayesian Optimization using the EI acquisition function
to find optimal reaction conditions (temperature, time, catalyst concentration)
that maximize reaction yield.

The goal is to understand how the optimization process affects the final
optimal conditions and yield.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from pyDOE import lhs

# -------------------------------------------------
# 1) Define the "True" Reaction Yield Function (Black Box)
# -------------------------------------------------
def true_yield(x):
    """
    Simulated reaction yield function.
    x: array of [temperature, time, concentration]
    Returns: yield (0-100)

    This function simulates a realistic chemical reaction with:
    - Optimal temperature around 80-90°C
    - Optimal reaction time around 60-80 min
    - Optimal catalyst concentration around 2.5-3.5 mM
    """
    x = np.atleast_2d(x)
    temp = x[:, 0]
    time = x[:, 1]
    conc = x[:, 2]

    # Normalize inputs to [0, 1] range for function calculation
    temp_norm = (temp - 30) / (110 - 30)
    time_norm = (time - 10) / (100 - 10)
    conc_norm = (conc - 0.835) / (4.175 - 0.835)

    # Complex yield surface with multiple local optima
    yield_value = (
        80 * np.exp(-((temp_norm - 0.7)**2 + (time_norm - 0.6)**2 + (conc_norm - 0.6)**2) / 0.1)
        + 60 * np.exp(-((temp_norm - 0.4)**2 + (time_norm - 0.3)**2 + (conc_norm - 0.3)**2) / 0.08)
        + 20 * np.sin(5 * temp_norm) * np.cos(3 * time_norm)
    )

    return np.clip(yield_value, 0, 100)


def noisy_yield(x, noise_std=2.0):
    """Noisy version of the yield function to simulate experimental uncertainty."""
    y = true_yield(x)
    y_flat = np.asarray(y).flatten()
    noise = noise_std * np.random.randn(len(y_flat))
    return y_flat + noise


# -------------------------------------------------
# 2) RBF Kernel for Gaussian Process
# -------------------------------------------------
def rbf_kernel(x1, x2, sigma_f=1.0, length_scale=1.0):
    """
    RBF (Squared Exponential) kernel for Gaussian Process.
    """
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)

    # Normalize inputs for kernel computation
    dists = cdist(x1, x2, metric='euclidean')
    sqdist = dists ** 2
    return sigma_f**2 * np.exp(-0.5 * sqdist / (length_scale**2))


# -------------------------------------------------
# 3) Gaussian Process Prediction
# -------------------------------------------------
def gaussian_process(X_train, y_train, X_test,
                     length_scale=1.0, sigma_f=1.0, noise_var=4.0):
    """
    GP regression with RBF kernel.
    """
    eps = 1e-10
    noise = noise_var + eps

    # Ensure y_train is 1D array
    y_train = np.asarray(y_train).flatten()

    K = rbf_kernel(X_train, X_train, sigma_f, length_scale) + noise * np.eye(len(X_train))
    K_inv = np.linalg.inv(K)

    K_s = rbf_kernel(X_train, X_test, sigma_f, length_scale)
    K_ss = rbf_kernel(X_test, X_test, sigma_f, length_scale) + noise * np.eye(len(X_test))

    alpha = K_inv @ y_train
    mu_s = K_s.T @ alpha

    cov_s = K_ss - K_s.T @ K_inv @ K_s
    std_s = np.sqrt(np.maximum(np.diag(cov_s), 0.0))
    return mu_s, std_s


# -------------------------------------------------
# 4) Expected Improvement (EI) Acquisition Function
# -------------------------------------------------
def expected_improvement(mu, sigma, f_max, xi=0.01):
    """
    Expected Improvement acquisition function.

    EI(x) = (mu(x) - f_max - xi) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (mu(x) - f_max - xi) / sigma(x)

    Parameters:
    - mu: GP posterior mean
    - sigma: GP posterior standard deviation
    - f_max: Current best observed value
    - xi: Exploration parameter (higher = more exploration)
    """
    Z = np.zeros_like(mu)
    mask = sigma > 0
    Z[mask] = (mu[mask] - f_max - xi) / sigma[mask]

    ei = (mu - f_max - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


# -------------------------------------------------
# 5) Bayesian Optimization with EI
# -------------------------------------------------
def bayesian_optimization_ei(n_iterations=15,
                             n_initial=5,
                             xi=0.01,
                             bounds=None,
                             noise_std=2.0,
                             verbose=True):
    """
    Run Bayesian Optimization using Expected Improvement.

    Parameters:
    - n_iterations: Number of optimization iterations
    - n_initial: Number of initial random samples
    - xi: EI exploration parameter
    - bounds: Parameter bounds [[temp_min, temp_max], [time_min, time_max], [conc_min, conc_max]]
    - noise_std: Standard deviation of measurement noise
    - verbose: Whether to print progress

    Returns:
    - results: Dictionary containing optimization history
    """
    if bounds is None:
        bounds = np.array([
            [30, 110],      # Temperature (°C)
            [10, 100],      # Time (min)
            [0.835, 4.175]  # Catalyst concentration (mM)
        ])

    n_params = bounds.shape[0]

    # Initialize with Latin Hypercube Sampling
    np.random.seed(42)
    lhs_samples = lhs(n_params, samples=n_initial)
    X_train = lhs_samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    y_train = noisy_yield(X_train, noise_std).flatten()

    # Storage for optimization history
    history = {
        'X': [X_train.copy()],
        'y': [y_train.copy()],
        'best_y': [np.max(y_train)],
        'best_X': [X_train[np.argmax(y_train)].copy()],
        'ei_max': []
    }

    if verbose:
        print("=" * 70)
        print("Bayesian Optimization with Expected Improvement (EI)")
        print("=" * 70)
        print(f"\nInitial samples (LHS): {n_initial}")
        print(f"Optimization iterations: {n_iterations}")
        print(f"EI exploration parameter (xi): {xi}")
        print(f"Observation noise std: {noise_std}")
        print("\n" + "-" * 70)
        print(f"{'Iter':<6} {'Temp':<10} {'Time':<10} {'Conc':<12} {'Yield':<10} {'Best':<10}")
        print("-" * 70)
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            print(f"{i+1:<6} {x[0]:<10.2f} {x[1]:<10.2f} {x[2]:<12.4f} {y:<10.2f} {np.max(y_train[:i+1]):<10.2f}")

    # Optimization loop
    length_scale = 0.5  # Initial hyperparameters
    sigma_f = 20.0      # Signal variance (yield scale)
    noise_var = noise_std ** 2

    for iteration in range(n_iterations):
        # Create a fine grid for prediction and EI calculation
        # (In practice, use optimization algorithm to find max EI)
        n_grid = 20
        temp_grid = np.linspace(bounds[0, 0], bounds[0, 1], n_grid)
        time_grid = np.linspace(bounds[1, 0], bounds[1, 1], n_grid)
        conc_grid = np.linspace(bounds[2, 0], bounds[2, 1], n_grid)

        T, Ti, C = np.meshgrid(temp_grid, time_grid, conc_grid, indexing='ij')
        X_test = np.column_stack([T.ravel(), Ti.ravel(), C.ravel()])

        # GP prediction
        mu, sigma = gaussian_process(X_train, y_train, X_test,
                                     length_scale, sigma_f, noise_var)

        # Calculate EI
        f_max = np.max(y_train)
        ei = expected_improvement(mu, sigma, f_max, xi)

        # Find next point (max EI)
        next_idx = np.argmax(ei)
        X_next = X_test[next_idx].reshape(1, -1)
        y_next = float(noisy_yield(X_next, noise_std)[0])

        # Update training data
        X_train = np.vstack([X_train, X_next])
        y_train = np.append(y_train, y_next)

        # Store history
        history['X'].append(X_train.copy())
        history['y'].append(y_train.copy())
        history['best_y'].append(np.max(y_train))
        history['best_X'].append(X_train[np.argmax(y_train)].copy())
        history['ei_max'].append(np.max(ei))

        if verbose:
            print(f"{iteration + n_initial + 1:<6} {X_next[0, 0]:<10.2f} {X_next[0, 1]:<10.2f} "
                  f"{X_next[0, 2]:<12.4f} {y_next:<10.2f} {history['best_y'][-1]:<10.2f}")

    if verbose:
        print("-" * 70)

    # Convert to arrays (only for items with consistent shapes)
    history['best_y'] = np.array(history['best_y'])
    history['best_X'] = np.array(history['best_X'])
    # history['X'] and history['y'] remain as lists due to varying shapes

    return history


# -------------------------------------------------
# 6) Visualization Functions
# -------------------------------------------------
def plot_optimization_progress(history, bounds):
    """Plot the optimization progress over iterations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Best yield vs iteration
    ax1 = axes[0, 0]
    iterations = np.arange(len(history['best_y']))
    ax1.plot(iterations, history['best_y'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Yield (%)', fontsize=12)
    ax1.set_title('Optimization Progress: Best Yield vs Iteration', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter evolution
    ax2 = axes[0, 1]
    best_X = history['best_X']
    ax2.plot(iterations, best_X[:, 0], 'r-o', label='Temperature (°C)', linewidth=2, markersize=5)
    ax2.plot(iterations, best_X[:, 1], 'g-s', label='Time (min)', linewidth=2, markersize=5)
    ax2.plot(iterations, best_X[:, 2] * 20, 'b-^', label='Concentration (mM) x20', linewidth=2, markersize=5)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Parameter Value', fontsize=12)
    ax2.set_title('Evolution of Optimal Parameters', fontsize=13)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: All sampled yields
    ax3 = axes[1, 0]
    all_y = history['y'][-1]
    colors = ['red' if i < 5 else 'blue' for i in range(len(all_y))]
    ax3.scatter(range(len(all_y)), all_y, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax3.axhline(y=np.max(all_y), color='green', linestyle='--', linewidth=2, label=f'Best: {np.max(all_y):.2f}%')
    ax3.axvline(x=4.5, color='gray', linestyle=':', linewidth=2, label='LHS / BO boundary')
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('Observed Yield (%)', fontsize=12)
    ax3.set_title('All Observed Yields (Red=LHS, Blue=BO)', fontsize=13)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: 2D projection of samples
    ax4 = axes[1, 1]
    all_X = history['X'][-1]  # Last entry (list)
    scatter = ax4.scatter(all_X[:, 0], all_X[:, 1], c=all_y, s=150, cmap='viridis',
                          edgecolors='black', linewidth=1, alpha=0.8)
    ax4.set_xlabel('Temperature (°C)', fontsize=12)
    ax4.set_ylabel('Time (min)', fontsize=12)
    ax4.set_title('Sample Distribution (Color = Yield)', fontsize=13)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Yield (%)', fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('P4Q4_optimization_progress.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_exploration_strategies():
    """
    Compare different exploration strategies by varying the xi parameter.
    """
    xi_values = [0.0, 0.01, 0.1, 1.0]
    results = {}

    print("\n" + "=" * 70)
    print("Comparing Different EI Exploration Parameters (xi)")
    print("=" * 70)
    print(f"\n{'xi':<10} {'Final Yield':<15} {'Temp':<10} {'Time':<10} {'Conc':<10}")
    print("-" * 70)

    for xi in xi_values:
        history = bayesian_optimization_ei(n_iterations=10, xi=xi, verbose=False)
        best_idx = np.argmax(history['y'][-1])
        best_X = history['X'][-1][best_idx]
        best_y = history['y'][-1][best_idx]
        results[xi] = {
            'history': history,
            'best_X': best_X,
            'best_y': best_y
        }
        print(f"{xi:<10.2f} {best_y:<15.2f} {best_X[0]:<10.2f} {best_X[1]:<10.2f} {best_X[2]:<10.4f}")

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Convergence curves
    ax1 = axes[0]
    colors = ['red', 'blue', 'green', 'orange']
    for (xi, result), color in zip(results.items(), colors):
        ax1.plot(result['history']['best_y'], '-o', color=color,
                label=f'xi = {xi}', linewidth=2, markersize=5)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Yield (%)', fontsize=12)
    ax1.set_title('Effect of Exploration Parameter on Convergence', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final optimal conditions
    ax2 = axes[1]
    xi_list = list(results.keys())
    temps = [results[xi]['best_X'][0] for xi in xi_list]
    times = [results[xi]['best_X'][1] for xi in xi_list]
    concs = [results[xi]['best_X'][2] for xi in xi_list]

    x_pos = np.arange(len(xi_list))
    width = 0.25

    # Normalize for visualization
    ax2.bar(x_pos - width, temps, width, label='Temperature (°C)', alpha=0.8)
    ax2.bar(x_pos, times, width, label='Time (min)', alpha=0.8)
    ax2.bar(x_pos + width, [c * 25 for c in concs], width, label='Concentration (mM) x25', alpha=0.8)

    ax2.set_xlabel('Exploration Parameter (xi)', fontsize=12)
    ax2.set_ylabel('Parameter Value', fontsize=12)
    ax2.set_title('Optimal Conditions vs Exploration Level', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{xi:.2f}' for xi in xi_list])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('P4Q4_exploration_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# -------------------------------------------------
# 7) Main Execution
# -------------------------------------------------
if __name__ == "__main__":

    # Run Bayesian Optimization with EI
    print("\n" + "=" * 70)
    print("TASK: Understanding Bayesian Optimization Process")
    print("=" * 70)
    print("""
    This demonstration shows how Bayesian Optimization with Expected
    Improvement (EI) finds optimal reaction conditions.

    Key Questions to Consider:
    1. How does the exploration parameter (xi) affect the search?
    2. How quickly does the algorithm converge to high-yield regions?
    3. How do the optimal conditions change with different xi values?
    4. What is the trade-off between exploration and exploitation?
    """)

    # Run single optimization
    history = bayesian_optimization_ei(n_iterations=15, n_initial=5, xi=0.01)

    # Print final results
    print("\n" + "=" * 70)
    print("FINAL OPTIMIZATION RESULTS")
    print("=" * 70)

    best_idx = np.argmax(history['y'][-1])
    best_X = history['X'][-1][best_idx]
    best_y = history['y'][-1][best_idx]

    print(f"\nOptimal Conditions:")
    print(f"  Temperature:          {best_X[0]:.2f} °C")
    print(f"  Reaction Time:        {best_X[1]:.2f} min")
    print(f"  Catalyst Concentration: {best_X[2]:.4f} mM")
    print(f"\nMaximum Observed Yield: {best_y:.2f}%")

    # Compare with true optimum
    true_opt = true_yield(best_X.reshape(1, -1))[0]
    print(f"True Yield at Optimum:  {true_opt:.2f}%")

    # Visualize optimization progress
    bounds = np.array([[30, 110], [10, 100], [0.835, 4.175]])
    plot_optimization_progress(history, bounds)

    # Compare different exploration strategies
    compare_exploration_strategies()

