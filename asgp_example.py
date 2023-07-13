import numpy as np
import matplotlib.pyplot as plt
from utils import gen_sine
from utils import plot_static
from scipy.optimize import minimize
from plot_utils import plot_gp, plot_gp_animation, plot_gp_dynamic
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

   # parameters
   n = 40  #number of data points
   n_test = 100
   mu_0= 0 #prior mean
   h_0 = 1 #initializaton of scaling param
   l_0 = [1] #initializaton of time-scale param

   #generate sine wave
   X_data = np.linspace(0,2*np.pi,n)
   X_data = X_data.reshape(n, 1)
   Y_data= gen_sine(X_data,f=1, mean=0)
   X_test = np.linspace(0,2*np.pi,n_test)
   X_test = X_test.reshape(n_test, 1)

   as_gp = Adaptive_Sparse_GPR()
   [mu, var] = as_gp.basic_gp(h_0, l_0, mu_0, Y_data, X_data, X_test)
   plot_static(X_data,Y_data,X_test,mu,var)



   #Compute
   print(Y_data)




