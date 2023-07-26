import numpy as np
import matplotlib.pyplot as plt
from utils import gen_sine
from utils import plot_static
from scipy.optimize import minimize
import os

from adaptive_sgp import Adaptive_Sparse_GPR

"""
This is a Python script that demonstrates Gaussian Process Regression example for a simple generated sine wave.

The script performs various steps, including data preprocessing, model training,
and visualization of the regression results.

Author: Hakim Amer
Date: May 15, 2023
"""


if __name__ == '__main__':
   as_gp = Adaptive_Sparse_GPR()

   # parameters
   n = 20  #number of data points
   n_ind =10
   n_test = 100
   mu_0= 0 #prior mean
   h_0 = 1 #initializaton of scaling param
   l_0 = [1] #initializaton of time-scale param
   σ = 0.01
   λ = 0.97
   delta = np.zeros((n,n))
   # Calculate Delta
   #for i in range (n):
   #    delta[i,i] =  λ**(n-i)


   #generate sine wave
   X_data = np.linspace(0,2*np.pi,n)
   X_data = X_data.reshape(n, 1)
   Y_data= gen_sine(X_data,f=1, mean=0)
   U = np.linspace(0,2*np.pi,n_ind)
   U = U.reshape(n_ind, 1)
   X_test = np.linspace(0,2*np.pi+1,n_test)
   X_test = X_test.reshape(n_test, 1)

   #[mu, var] = as_gp.basic_gp(h_0, l_0, mu_0, Y_data, X_data, X_test)
   #plot_static(X_data,Y_data,X_test,mu,var)

    #Implementation of AGP
    # compute new mu_λs, var_λs
   x_t = X_data[len(X_data) - 1] + 0.01
   x_test = X_test[len(X_test) - 1] + 0.01

   y_t = gen_sine(x_t, f=1, mean=0)
   U_n = U
   U_n = U
   for i in range(100):
       x_t = x_t + 0.1
       x_test = x_test + 0.01
       y_t =gen_sine(x_t ,f=1, mean=0)
       X_test = np.append(X_test, x_test)
       X_test = X_test.reshape(len(X_test), 1)
       [mu_λs, var_λs] = as_gp.fast_adaptive_gp( y_t, x_t, X_data,Y_data,X_test,U, h_0,l_0,mu_0,σ,λ,delta)
       U_n =  np.append(U_n, x_t)

       Y_U =  gen_sine(U_n, f=1, mean=0)
       plot_static(X_data, Y_data, X_test, mu_λs, var_λs, U_n, Y_U)



