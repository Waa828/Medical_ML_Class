from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def mbgd_function(J: Callable,
                 dJ: Callable,
                 X: np.ndarray,
                 y: np.ndarray,
                 step_size: float,
                 epsilon: float,
                 batch_size: int = 32,
                 max_epoch: int = 100,
                 *arg):
    num = len(X)
    dim = len(X[0])
    theta = np.zeros([dim])
    history = []
    
    for i in range(max_epoch):
        indices = np.random.permutation(num)
        num_batches = num // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            dobj = dJ(theta, X_batch, y_batch)
            theta = theta - dobj * step_size
            
            loss = J(theta, X, y)
            history.append(loss)
        
        if i > 0 and abs(history[-1] - history[-(num_batches+1)]) <= epsilon:
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
    P2Q2: Mini-Batch Gradient Descent (MBGD) for Linear Regression
    
    Objective: Fit a linear model using MBGD (updates after each mini-batch)
    Dataset: p2_fittingdata_x.txt (features), p2_fittingdatap2_y.txt (labels)
    
    Comparison with BGD and SGD:
        - BGD: uses all samples per update
        - SGD: uses 1 sample per update
        - MBGD: uses batch_size samples per update (balance between stability and speed)
    
    Hyperparameters:
        - step_size: 0.0001
        - epsilon: 0.5
        - batch_size: 32
        - max_epoch: 100
    
    Output: P2Q3_plot.png showing RMSE vs iterations with epoch markers
    """
    X = np.loadtxt("p2_fittingdata_x.txt")
    y = np.loadtxt("p2_fittingdatap2_y.txt")
    
    theta, history = mbgd_function(J, dJ, X, y, 0.0001, 0.5, batch_size=32, max_epoch=100)
    print(theta)
    num = len(X)
    batch_size = 32
    num_batches = num // batch_size
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history)+1), history, 'g-', linewidth=0.5)
    plt.scatter(range(1, len(history)+1), history, c='blue', s=5, zorder=5)
    epoch_markers = [i * num_batches for i in range(1, len(history) // num_batches + 1)]
    epoch_markers = [m for m in epoch_markers if m <= len(history)]
    plt.scatter(epoch_markers, [history[m-1] for m in epoch_markers], 
                marker='*', c='gold', s=150, edgecolors='black', zorder=10, label='Epoch end')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (RMSE)')
    plt.title('Mini-Batch Gradient Descent (batch_size=32)')
    plt.legend()
    plt.grid(True)
    plt.savefig('P2Q3_plot.png', dpi=150)
    plt.show()
