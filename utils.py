import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

"""
Python script for visualisation of GPR

Author: Hakim Amer
Date: May 15, 2023
"""

def gen_sine(x,f,mean):

    y = np.sin(f*x) + mean

    return y



def plot_static(X_data,Y_data,X_test,mu,var):

    plt.scatter(X_data, Y_data, color='red', label='Data')
    plt.plot(X_test, mu, color='blue', label='Prediction')
    plt.fill_between(
        X_test[:,0],
        mu[:,0] - np.sqrt(np.diag(var)),
        mu[:,0] + np.sqrt(np.diag(var)),
        color='gray',
        alpha=0.4,
        label='Uncertainty'
    )
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.show()
    return