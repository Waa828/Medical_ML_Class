import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
from PIL import Image

def gd_function_with_history(
    obj: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    init_guess: np.ndarray,
    step_size: float,
    epsilon: float,
    max_iter: int = 100
) -> tuple[np.ndarray, dict]:
    x = init_guess
    history = {'iter': [], 'grad_norm': [], 'x_vals': []}
  
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
            break
    else:
        print('Reached the maximum number of convergence steps')
    opt_point = x
    
    return opt_point, history


def Numerical_gd_function_with_history(
    obj: Callable[[np.ndarray], float],
    h: float, 
    init_guess: np.ndarray,
    step_size: float,
    epsilon: float,
    max_iter: int = 100,
    grad_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> tuple[np.ndarray, dict]:
    x = init_guess
    history = {'iter': [], 'grad_norm': [], 'x_vals': [], 'analytical_grad_norm': [], 'grad_diff': []}
  
    for i in range(max_iter):
        y = x
        dobj = compute_numerical_gradient(obj, x, h)
        grad_norm = float(np.linalg.norm(dobj))
        history['iter'].append(i+1)
        history['grad_norm'].append(grad_norm)
        history['x_vals'].append(x.copy())
        
        if grad_func is not None:
            analytical_grad = grad_func(x)
            analytical_grad_norm = float(np.linalg.norm(analytical_grad))
            grad_diff = float(np.linalg.norm(dobj - analytical_grad))
            history['analytical_grad_norm'].append(analytical_grad_norm)
            history['grad_diff'].append(grad_diff)
        
        x = x - dobj * step_size
        if abs(float(obj(y) - obj(x))) <= epsilon:
            print(f"converged after {i+1} steps")
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


def compute_numerical_gradient(obj: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i, 0] += h
        x_minus[i, 0] -= h
        grad[i, 0] = (obj(x_plus) - obj(x_minus)) / (2 * h)
    return grad


def compare_gradients_at_points(obj: Callable, grad_func: Callable, points: list, h: float = 1e-6):
    print("\n" + "="*70)
    print("Gradient Comparison: Analytical vs Numerical")
    print("="*70)
    print(f"{'Point':<30} {'Analytical':<20} {'Numerical':<20} {'Difference':<15}")
    print("-"*70)
    
    for i, point in enumerate(points):
        x = np.array([[point[0]], [point[1]]])
        analytical_grad = grad_func(x)
        numerical_grad = compute_numerical_gradient(obj, x, h)
        diff = float(np.linalg.norm(analytical_grad - numerical_grad))
        
        print(f"Point {i+1} ({point[0]:.2f}, {point[1]:.2f}): [{analytical_grad[0,0]:.6e}, {analytical_grad[1,0]:.6e}] -> [{numerical_grad[0,0]:.6e}, {numerical_grad[1,0]:.6e}] -> {diff:.6e}")
    
    print("="*70 + "\n")

def plot_results(history, u):
    has_comparison = 'analytical_grad_norm' in history and len(history['analytical_grad_norm']) > 0
    
    if has_comparison:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(history['iter'], history['grad_norm'], 'b-', linewidth=2, label='Numerical')
        axes[0].plot(history['iter'], history['analytical_grad_norm'], 'r--', linewidth=2, label='Analytical')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Gradient Norm')
        axes[0].set_title('Gradient Norm: Analytical vs Numerical')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['iter'], history['grad_diff'], 'g-', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Gradient Difference')
        axes[1].set_title('Norm Difference (Analytical - Numerical)')
        axes[1].grid(True, alpha=0.3)
        
        contour_ax = axes[2]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history['iter'], history['grad_norm'], 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Gradient Norm')
        axes[0].set_title('Gradient Norm vs Iteration')
        axes[0].grid(True, alpha=0.3)
        
        contour_ax = axes[1]
    
    x = np.linspace(-5, 25, 100)
    y = np.linspace(-5, 25, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([[X[i, j]], [Y[i, j]]])
            Z[i, j] = neg_gaussian(point) * 1e6
    
    contour = contour_ax.contour(X, Y, Z, levels=20, cmap='viridis')
    
    x_vals = np.array(history['x_vals'])
    optimal_point = x_vals[-1]
    start_point = x_vals[0]
    
    contour_ax.scatter([u[0,0]], [u[1,0]], c='red', s=80, marker='*', label='Target', zorder=5)
    contour_ax.scatter([optimal_point[0,0]], [optimal_point[1,0]], c='green', s=50, marker='s', label='Optimal', zorder=5)
    contour_ax.scatter([start_point[0,0]], [start_point[1,0]], c='blue', s=50, marker='o', label='Start', zorder=5)
    
    contour_ax.plot(x_vals[:, 0, 0], x_vals[:, 1, 0], 'r-', linewidth=1.5, alpha=0.7)
    contour_ax.scatter(x_vals[:, 0, 0], x_vals[:, 1, 0], c='red', s=10, alpha=0.5, label='Training Path')
    
    contour_ax.set_xlabel('X')
    contour_ax.set_ylabel('Y')
    contour_ax.set_title('Gaussian Contour with Training Path')
    contour_ax.legend(loc='upper left')
    
    cbar = plt.colorbar(contour, ax=contour_ax, shrink=0.8)
    cbar.set_label('$*10^{-6}$')
    
    plt.tight_layout()
    plt.savefig('P1Q3_gradient_comparison.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    """
    P1Q3: Numerical vs Analytical Gradient Comparison
    
    Objective: Verify numerical gradient computation against analytical gradient
    Target function: 2D Gaussian (same as P1Q21)
    
    Workflow:
        1. Run gradient descent using numerical gradients
        2. Compare gradients at multiple points during optimization
        3. Visualize convergence and gradient differences
    
    Hyperparameters: step_size=1e6, epsilon=1e-15, max_iter=100, h=1e-6
    Output: Convergence plots with analytical vs numerical comparison
    """
    u = np.array([[10.], [10.]])
    result, history = Numerical_gd_function_with_history(neg_gaussian, 1e-6, np.array([[0.], [0.]]), 1000000, 1e-15, 100, grad_neg_gaussian)
    print("Optimal point:", result.T)
    print("Function value at optimal point:", neg_gaussian(result))
    
    x_vals = np.array(history['x_vals'])
    n = len(x_vals)
    test_points = [
        [x_vals[0][0,0], x_vals[0][1,0]],
        [x_vals[n//6][0,0], x_vals[n//6][1,0]],
        [x_vals[n//3][0,0], x_vals[n//3][1,0]],
        [x_vals[n//2][0,0], x_vals[n//2][1,0]],
        [x_vals[2*n//3][0,0], x_vals[2*n//3][1,0]],
        [x_vals[5*n//6][0,0], x_vals[5*n//6][1,0]],
        [x_vals[-1][0,0], x_vals[-1][1,0]]
    ]
    compare_gradients_at_points(neg_gaussian, grad_neg_gaussian, test_points, 1e-6)
    
    plot_results(history, u)
