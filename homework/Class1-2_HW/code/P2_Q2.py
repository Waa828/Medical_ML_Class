from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def sgd_function(J: Callable,
                 dJ: Callable,
                 X: np.ndarray,
                 y: np.ndarray,
                 step_size: float,
                 epsilon: float,
                 max_epoch: int = 100,
                 *arg):
    num = len(X)
    dim = len(X[0])
    theta = np.zeros([dim])
    history = []
    
    for i in range(max_epoch):
        indices = np.random.permutation(num)
        for j, idx in enumerate(indices):
            xi = X[idx:idx+1]
            yi = y[idx]
            dobj = dJ(theta, xi, yi)
            theta = theta - dobj * step_size
            
            loss = J(theta, X, y)
            history.append(loss)
        
        if i > 0 and abs(history[-1] - history[-num-1]) <= epsilon:
            print("converged after %d epochs" % (i+1))
            break
    else:
        print('Reached the maximum number of convergence steps')
    
    return theta, history
    
def J(theta: np.ndarray,
       x: np.ndarray,
       y: np.ndarray):
    predictions = x @ theta
    return np.sqrt(np.mean((predictions - y) ** 2))

def dJ(theta: np.ndarray,
       x: np.ndarray,
       y: np.ndarray):
    n = len(x)
    x_T = np.transpose(x)
    return (2/n) * np.sum((theta @ x_T - y)[:, np.newaxis] * x, axis=0)

if __name__ == '__main__':
    """
    P2Q2: Stochastic Gradient Descent (SGD) for Linear Regression
    
    Objective: Fit a linear model using SGD (updates after each sample)
    Dataset: p2_fittingdata_x.txt (features), p2_fittingdatap2_y.txt (labels)
    
    Comparison with BGD:
        - BGD: computes gradient using all samples per epoch
        - SGD: computes gradient using one sample at a time
    
    Hyperparameters:
        - step_size: 0.0001
        - epsilon: 0.5
        - max_epoch: 100
    
    Output: P2Q2_plot.png showing RMSE vs iterations with epoch markers
    """
    X = np.loadtxt("p2_fittingdata_x.txt")
    y = np.loadtxt("p2_fittingdatap2_y.txt")
    
    theta, history = sgd_function(J, dJ, X, y, 0.0001, 0.5, 100)
    print(theta)
    num = len(X)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history)+1), history, 'r-', linewidth=0.5)
    plt.scatter(range(1, len(history)+1), history, c='blue', s=5, zorder=5)
    epoch_markers = [i * num for i in range(1, len(history) // num + 1)]
    epoch_markers = [m for m in epoch_markers if m <= len(history)]
    plt.scatter(epoch_markers, [history[m-1] for m in epoch_markers], 
                marker='*', c='gold', s=150, edgecolors='black', zorder=10, label='Epoch end')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (RMSE)')
    plt.title('Stochastic Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.savefig('P2Q2_plot.png', dpi=150)
    plt.show()