import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
"""
Python script for visualisation of GPR

Author: Hakim Amer
Date: May 15, 2023
"""

def gen_sine(x,f,mean):

    y = np.sin(f*x) + mean + 0.5*np.cos(2*f*x) + 0.1*np.sin(10*f*x)

    return y


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
iteration= -1
def plot_static(X_data,Y_data,X_test,mu,var, U_n, Y_U,x_t,y_t,color_p,g_t, name,clear_plt,moving):

    #plt.title(name)


    ax1.scatter(X_data, Y_data, color='red', label='Data')
    ax1.scatter(U_n, U_n*0.0, color='pink', label='pseudo-points', marker ='x')
    ax1.scatter(x_t, y_t,[150], color='purple', label='new data', marker ='*')

    ax1.scatter(X_test, g_t, [50], color='cyan', label='gt', marker='x')

    ax1.plot(X_test, mu[:,0], color=color_p, label=name)
    ax1.fill_between(
         X_test[:,0],
         mu[:,0] - np.sqrt(np.diag(var)),
         mu[:,0] + np.sqrt(np.diag(var)),
         color=color_p,
         alpha=0.13,
         label='Uncertainty'
     )

    ax1.set_xlabel('t')
    ax1.set_ylabel('f(t)')
    plt.legend()
    ax1.set_xlim([ X_test[len(X_test)-1]-2*np.pi,X_test[len(X_test)-1]+1,])
    ax1.set_xlim([0.0, 2 * np.pi])

    if moving==1:
       ax1.set_xlim([ x_t-1*np.pi,X_test[len(X_test)-1]+0.5])
       ax1.set_ylim([-4.1, 4.2])
    #ax1.set_xlim([0.0, 2*np.pi ])
    #ax2.set_ylim([-0.3,0.3 ])

    #errors = g_t - mu
    #ax2.plot(X_test, errors, color_p, label='Prediction Error')
    #ax2.set_xlabel('t')
    #ax2.set_ylabel('Prediction Error')
    #ax2.set_xlim([X_test[len(X_test) - 1] - 2 * np.pi, X_test[len(X_test) - 1] + 1, ])
    #ax2.legend()
    plt.tight_layout()
    # Display the plot
    #plt.show(block=False)
    plt.show(block=False)
    plt.pause(1)
    if clear_plt==1:
       ax1.clear()
    return
