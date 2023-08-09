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



def plot_static(X_data,Y_data,X_test,mu,var, U_n, Y_U,x_t,y_t):

    plt.scatter(X_data, Y_data, color='red', label='Data')
    plt.scatter(U_n, Y_U, color='green', label='pseudo-points', marker ='x')
    plt.scatter(x_t, y_t, color='pink', label='new data', marker ='x')

    plt.plot(X_test, mu[:,0], color='blue', label='Prediction')
    # plt.fill_between(
    #     X_test[:,0],
    #     mu[:,0] - np.sqrt(np.diag(var)),
    #     mu[:,0] + np.sqrt(np.diag(var)),
    #     color='gray',
    #     alpha=0.4,
    #     label='Uncertainty'
    # )
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.clf()
   #plt.show()
    return
