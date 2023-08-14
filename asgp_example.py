import numpy as np
import matplotlib.pyplot as plt
from utils import gen_sine
from utils import plot_static
from scipy.optimize import minimize
import os
import random
from adaptive_sgp import Adaptive_Sparse_GPR

"""
This is a Python script that demonstrates Gaussian Process Regression example for a simple generated sine wave.

The script performs various steps, including data preprocessing, model training,
and visualization of the regression results.

Author: Hakim Amer
Date: May 15, 2023
"""


if __name__ == '__main__':


   # parameters
   n = 8 #number of data points
   n_ind =5
   n_test = 1000
   mu_0= 0 #prior mean
   h_0 = 1 #initializaton of scaling param
   l_0 = [1] #initializaton of time-scale param
   σ = 0.0001
   λ = 0.9
   R_th = 0.0001
   delta = np.zeros((n,n))
   for j in range(n):
       delta[j, j] = λ ** (n - j)


   #generate sine wave
   X_data = np.linspace(0,2*np.pi,n)
   X_data = X_data.reshape(n, 1)
   Y_data= gen_sine(X_data,f=1, mean=0)
   U = np.linspace(0,2*np.pi,n_ind)
   U = U.reshape(n_ind, 1)
   X_test = np.linspace(0.0,2*np.pi+1,n_test)

   X_test = X_test.reshape(n_test, 1)



    #Implementation of AGP
    # compute new mu_λs, var_λs
   x_t = X_data[len(X_data) - 1] + 0.01
   x_test = X_test[len(X_test) - 1] + 0.1

   y_t = gen_sine(x_t, f=1, mean=0)
   U_n = U
   U_n = U
   count =1
   dx = 0.2


   #print('delta', delta)
   U_0 = U
   as_gp = Adaptive_Sparse_GPR(h_0, l_0, X_data, Y_data, U_0, X_test,σ, λ, delta,R_th)
   for it in range(100):
       x_t = x_t + dx
       x_t = random.uniform(0, 2*np.pi)

       x_test = x_test  + dx
       y_t =gen_sine(x_t ,f=1, mean=0)
       x_t = np.array([x_t])
       x_t = x_t.reshape(1, 1)
       y_t = np.array([y_t])
       y_t = y_t.reshape(1, 1)

       #X_test = np.append(X_test, x_test)
       #X_test = np.delete(X_test, 0)

       #X_test = X_test.reshape(len(X_test), 1)
       #X_test = x_t +1


       [mu_λs, var_λs] = as_gp.fast_adaptive_gp(y_t, x_t, X_test, it)

       Y_U = np.zeros(len(U_n))    # calculate Y values of inducing points for visualisation
       Y_U = gen_sine(U_n, f=1, mean=0)
       print(len(X_test))
       print(len(mu_λs))
       plot_static(X_data, Y_data, X_test, mu_λs, var_λs, U_n, Y_U,x_t,y_t)
   plt.show()


