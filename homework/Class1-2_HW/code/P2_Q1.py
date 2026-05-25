from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def bgd_function(J:  Callable,
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
        theta_once = theta
        dobj = dJ(theta, X, y)
        theta = theta - dobj * step_size
        loss = J(theta, X, y)
        history.append(loss)
        
        if abs(loss - J(theta_once, X, y))<=epsilon:
            print("converged after %d steps" %(i+1))
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
    P2Q1: Batch Gradient Descent (BGD) for Linear Regression
    
    Objective: Fit a linear model to the provided data using BGD
    Dataset: p2_fittingdata_x.txt (features), p2_fittingdatap2_y.txt (labels)
    
    Hyperparameters:
        - step_size: 0.0001
        - epsilon: 0.5 (convergence threshold)
        - max_epoch: 100
    
    Output: P2Q1_plot.png showing RMSE vs iterations
    """
    X = np.loadtxt("p2_fittingdata_x.txt")
    y = np.loadtxt("p2_fittingdatap2_y.txt")
    
    theta, history = bgd_function(J, dJ, X, y, 0.0001, 0.5, 100)
    print(theta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history)+1), history, 'b-o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (RMSE)')
    plt.title('Batch Gradient Descent')
    plt.grid(True)
    plt.savefig('P2Q1_plot.png', dpi=150)
    plt.show()