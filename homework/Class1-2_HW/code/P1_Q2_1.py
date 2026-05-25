import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from PIL import Image
import csv

def gd_function_with_history(
    obj: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    init_guess: np.ndarray,
    step_size: float,
    epsilon: float,
    max_iter: int = 100
) -> tuple[np.ndarray, dict]:
    x = init_guess.copy()
    history = {'iter': [], 'grad_norm': [], 'x_vals': [], 'converged': False}

    for i in range(max_iter):
        y = x
        dobj = grad(x)
        grad_norm = float(np.linalg.norm(dobj))
        history['iter'].append(i+1)
        history['grad_norm'].append(grad_norm)
        history['x_vals'].append(x.copy())

        x = x - dobj * step_size
        if abs(float(obj(y) - obj(x))) <= epsilon:
            print(f"converged after {i+1} steps")
            history['converged'] = True
            break
    else:
        print('Reached the maximum number of convergence steps')
    opt_point = x

    return opt_point, history

def neg_gaussian(x: np.ndarray, 
                 u: np.ndarray = np.array([[10.], [10.]]), 
                 sigma: np.ndarray = np.diag([1000., 1000.]), 
                 n: int = 2) -> float:
    det_Sigma = np.linalg.det(sigma)
    norm_factor = 1 / ((2 * np.pi) ** (n/2) * det_Sigma ** 0.5)
    diff = x - u
    sigma_inv = np.linalg.inv(sigma)
    exponent = -0.5 * (diff.T @ sigma_inv @ diff)[0, 0]
    return -norm_factor * np.exp(exponent)

def grad_neg_gaussian(x: np.ndarray,
                      u: np.ndarray = np.array([[10.], [10.]]), 
                      sigma: np.ndarray = np.diag([1000., 1000.]),
                      n: int = 2) -> np.ndarray:
    fx = neg_gaussian(x, u, sigma, n)
    sigma_inv = np.linalg.inv(sigma)
    return -fx * sigma_inv @ (x - u)

def plot_results(history, u):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['iter'], history['grad_norm'], 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Gradient Norm')
    axes[0].set_title('Gradient Norm vs Iteration')
    axes[0].grid(True, alpha=0.3)
    
    x = np.linspace(-5, 25, 100)
    y = np.linspace(-5, 25, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([[X[i, j]], [Y[i, j]]])
            Z[i, j] = neg_gaussian(point) * 1e6
    
    contour = axes[1].contour(X, Y, Z, levels=20, cmap='viridis')
    
    x_vals = np.array(history['x_vals'])
    optimal_point = x_vals[-1]
    start_point = x_vals[0]
    
    axes[1].scatter([u[0,0]], [u[1,0]], c='red', s=80, marker='*', label='Target', zorder=5)
    axes[1].scatter([optimal_point[0,0]], [optimal_point[1,0]], c='green', s=50, marker='s', label='Optimal', zorder=5)
    axes[1].scatter([start_point[0,0]], [start_point[1,0]], c='blue', s=50, marker='o', label='Start', zorder=5)
    
    axes[1].plot(x_vals[:, 0, 0], x_vals[:, 1, 0], 'r-', linewidth=1.5, alpha=0.7)
    axes[1].scatter(x_vals[:, 0, 0], x_vals[:, 1, 0], c='red', s=10, alpha=0.5, label='Training Path')
    
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Gaussian Contour with Training Path')
    axes[1].legend(loc='upper left')
    
    cbar = plt.colorbar(contour, ax=axes[1], shrink=0.8)
    cbar.set_label('$*10^{-6}$')
    
    plt.tight_layout()
    plt.savefig('P1Q21_plot_results.png', dpi=150)
    plt.show()

def analyze_initial_guess(u):
    """Analyze the effect of initial guess on convergence"""
    init_guesses = [
        (np.array([[0.], [0.]]), '(0, 0)'),
        (np.array([[5.], [5.]]), '(5, 5)'),
        (np.array([[15.], [15.]]), '(15, 15)'),
        (np.array([[-5.], [20.]]), '(-5, 20)'),
        (np.array([[20.], [-5.]]), '(20, -5)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Draw contour
    x = np.linspace(-5, 25, 100)
    y = np.linspace(-5, 25, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([[X[i, j]], [Y[i, j]]])
            Z[i, j] = neg_gaussian(point) * 1e6
    axes[1].contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
    axes[1].scatter([u[0,0]], [u[1,0]], c='red', s=100, marker='*', label='Target', zorder=5)

    colors = plt.cm.tab10(np.linspace(0, 1, len(init_guesses)))

    for (init_guess, label), color in zip(init_guesses, colors):
        _, history = gd_function_with_history(neg_gaussian, grad_neg_gaussian, init_guess, 1000000, 1e-15, 100)
        axes[0].plot(history['iter'], history['grad_norm'], '-', color=color, label=f'Init {label}', linewidth=2)

        x_vals = np.array(history['x_vals'])
        axes[1].plot(x_vals[:, 0, 0], x_vals[:, 1, 0], 'o-', color=color, markersize=3, alpha=0.7)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Gradient Norm')
    axes[0].set_title('Effect of Initial Guess on Convergence')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Optimization Paths from Different Initial Guesses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('P1Q21_initial_guess_analysis.png', dpi=150)
    plt.show()

def analyze_step_size(u):
    """Analyze the effect of step size on convergence"""
    step_sizes = [100, 1000, 10000, 100000, 1000000, 10000000, 20000000]
    init_guess = np.array([[0.], [0.]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(step_sizes)))
    results = []

    for step_size, color in zip(step_sizes, colors):
        result, history = gd_function_with_history(neg_gaussian, grad_neg_gaussian, init_guess, step_size, 1e-15, 100)
        distance = np.linalg.norm(result - u)
        results.append((step_size, len(history['iter']), distance, history['converged']))
        axes[0].plot(history['iter'], history['grad_norm'], '-', color=color, label=f'α={step_size:.0e}', linewidth=2)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Gradient Norm')
    axes[0].set_title('Effect of Step Size on Convergence Rate')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x_pos = np.arange(len(step_sizes))
    width = 0.35
    iters = [r[1] if r[3] else float('inf') for r in results]
    distances = [r[2] for r in results]

    ax2 = axes[1]
    bars1 = ax2.bar(x_pos - width/2, iters, width, label='Iterations', color='steelblue', alpha=0.7)
    ax2.set_ylabel('Iterations', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')

    ax3 = ax2.twinx()
    bars2 = ax3.bar(x_pos + width/2, distances, width, label='Distance to Target', color='coral', alpha=0.7)
    ax3.set_ylabel('Distance to Extremum', color='coral')
    ax3.tick_params(axis='y', labelcolor='coral')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{s:.0e}' for s in step_sizes])
    ax2.set_xlabel('Step Size (α)')
    ax2.set_title('Iterations and Distance to Extremum vs Step Size')

    plt.tight_layout()
    plt.savefig('P1Q21_step_size_analysis.png', dpi=150)
    plt.show()

    print("\nStep Size Analysis Results:")
    print("-" * 60)
    print(f"{'Step Size':>12} {'Iterations':>12} {'Distance to Extremum':>20}")
    print("-" * 60)
    for step_size, iters, dist, converged in results:
        iter_str = str(iters) if converged else 'inf'
        print(f"{step_size:>12.0e} {iter_str:>12} {dist:>20.6f}")

def analyze_epsilon(u):
    """Analyze the effect of convergence criterion on final solution"""
    epsilons = [1e-10, 1e-12, 1e-14, 1e-16, 1e-18]
    init_guess = np.array([[0.], [0.]])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = plt.cm.plasma(np.linspace(0, 1, len(epsilons)))
    results = []

    for epsilon, color in zip(epsilons, colors):
        result, history = gd_function_with_history(neg_gaussian, grad_neg_gaussian, init_guess, 1000000, epsilon, 200)
        error = np.linalg.norm(result - u)
        results.append((epsilon, len(history['iter']), error, result, history))

    x = np.linspace(9.95, 10.05, 100)
    y = np.linspace(9.95, 10.05, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([[X[i, j]], [Y[i, j]]])
            Z[i, j] = neg_gaussian(point) * 1e6
    
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    ax.scatter([u[0,0]], [u[1,0]], c='red', s=100, marker='*', label='Target', zorder=5)
    
    markers = ['o', 's', '^', 'D', 'v']
    for i, (epsilon, iters, error, result, history) in enumerate(results):
        ax.scatter([result[0,0]], [result[1,0]], c=[colors[i]], s=80, marker=markers[i], 
                       label=f'ε={epsilon:.0e}', zorder=4, edgecolors='black')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Effect of Convergence Criterion on Final Solution')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('P1Q21_epsilon_analysis.png', dpi=150)
    plt.show()

    print("\nConvergence Criterion Analysis Results:")
    print("-" * 60)
    print(f"{'ε':>12} {'Iterations':>12} {'Distance to Extremum':>20}")
    print("-" * 60)
    for epsilon, iters, dist, result, history in results:
        print(f"{epsilon:>12.0e} {iters:>12} {dist:>20.6f}")

def export_results_to_csv(results: list, filename: str):
    """Export results to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['init_guess_x', 'init_guess_y', 'step_size', 'epsilon', 'final_x', 'final_y', 'iterations'])
        for row in results:
            writer.writerow(row)
    print(f"Results exported to {filename}")

def run_parameter_sweep_and_export(obj, grad, init_guesses, step_sizes, epsilons, max_iter, filename):
    """Run parameter sweep and export results to CSV"""
    results = []
    total = len(init_guesses) * len(step_sizes) * len(epsilons)
    count = 0

    for init_guess in init_guesses:
        for step_size in step_sizes:
            for epsilon in epsilons:
                count += 1
                result, history = gd_function_with_history(obj, grad, init_guess, step_size, epsilon, max_iter)

                # Iterations: infinity if not converged
                iterations = len(history['iter']) if history['converged'] else float('inf')

                results.append([
                    f"{init_guess[0,0]:.6f}",
                    f"{init_guess[1,0]:.6f}",
                    f"{step_size:.6e}",
                    f"{epsilon:.6e}",
                    f"{result[0,0]:.6f}",
                    f"{result[1,0]:.6f}",
                    'inf' if iterations == float('inf') else int(iterations)
                ])

                if count % 10 == 0 or count == total:
                    print(f"Progress: {count}/{total}")

    export_results_to_csv(results, filename)
    return results

if __name__ == '__main__':
    """
    P1Q21: Gaussian Function Optimization with Parameter Sensitivity Analysis
    
    Objective: Find the maximum (minimum of negative) of 2D Gaussian function
    Target: u = [10, 10], sigma = diag([1000, 1000])
    
    Workflow:
        1. Initial optimization run from (0, 0)
        2. Analyze effect of initial guess on convergence
        3. Analyze effect of step size on convergence rate
        4. Analyze effect of epsilon on final solution accuracy
        5. Parameter sweep across all combinations and export to CSV
    
    Output: PNG plots and P1Q21_results.csv
    """
    u = np.array([[10.], [10.]])

    # Initial run
    print("=" * 60)
    print("Initial Optimization Run")
    print("=" * 60)
    result, history = gd_function_with_history(neg_gaussian, grad_neg_gaussian, np.array([[0.], [0.]]), 1000000, 1e-15, 100)
    print("Optimal point:", result.T)
    print("Function value at optimal point:", neg_gaussian(result))
    plot_results(history, u)

    # Parameter sensitivity analysis
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis")
    print("=" * 60)

    print("\n1. Analyzing initial guess effect...")
    analyze_initial_guess(u)

    print("\n2. Analyzing step size effect...")
    analyze_step_size(u)

    print("\n3. Analyzing convergence criterion effect...")
    analyze_epsilon(u)

    # Parameter sweep and CSV export
    print("\n" + "=" * 60)
    print("Parameter Sweep and CSV Export")
    print("=" * 60)

    init_guesses = [
        np.array([[0.], [0.]]),
        np.array([[5.], [5.]]),
        np.array([[15.], [15.]]),
        np.array([[-5.], [20.]]),
        np.array([[20.], [-5.]]),
    ]
    step_sizes = [100, 1000, 10000, 100000, 1000000, 5000000]
    epsilons = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25]

    run_parameter_sweep_and_export(
        neg_gaussian, grad_neg_gaussian,
        init_guesses, step_sizes, epsilons,
        max_iter=100,
        filename='P1Q21_results.csv'
    )
