import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import math

def gd_function(
    obj: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    init_guess: np.ndarray,
    step_size: float,
    epsilon: float,
    max_iter: int = 100
) -> np.ndarray:
    """
    Gradient Descent Function to minimize an objective function.

    Parameters:
    - obj: Objective function to minimize (must take a vector
      as input and return a scalar).
    - grad: Gradient function of the objective function
      (returns a vector of gradients).
    - init_guess: Initial guess for the parameters (a vector).
    - step_size: The step size (learning rate) for the gradient descent updates.
    - epsilon: The stopping criterion (the algorithm stops
      when the difference in objective function value on two
      successive steps is below a threshold).
    - max_iter: Maximum number of iterations to prevent infinite loops.

    Returns:
    - opt_point: The vector of parameters that minimize the objective function.
    """
    x = init_guess
  
    for i in range(max_iter):
        y = x
        dobj = grad(x)
        x = x - dobj * step_size
        if abs(obj(y) - obj(x))<=epsilon:
            print("converged after %d steps" %(i+1))
            break
    else:
            print('Reached the maximum number of convergence steps')
    opt_point = x

    return opt_point

def obj_test(x):
    y = math.sqrt(x)
    return y 

def grad_test(x):
    y = 1 / (2 * math.sqrt(x))
    return y

if __name__ == '__main__':
    """
    P1Q1: Gradient Descent Algorithm Test
    
    Objective: Use gradient descent to find the minimum of f(x) = sqrt(x)
    Initial guess: x0 = 100
    Hyperparameters: step_size=0.1, epsilon=0.1, max_iter=100
    
    Output: Optimal x value after convergence
    """
    result = gd_function(obj_test, grad_test, 100, 1, 0.000000001, 100)
    print(result)