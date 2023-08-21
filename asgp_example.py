import numpy as np
import matplotlib.pyplot as plt
from utils import gen_sine
from utils import plot_static
from scipy.optimize import minimize
import os
import random
from adaptive_sgp import Adaptive_Sparse_GPR
import time
"""
This is a Python script that demonstrates Gaussian Process Regression example for a simple generated sine wave.

The script performs various steps, including data preprocessing, model training,
and visualization of the regression results.

Author: Hakim Amer
Date: May 15, 2023
"""


if __name__ == '__main__':


   # parameters
   n_cycles =2
   n = 40#number of data points
   n_ind =int(n/2)
   #n_ind =5
   t_start= 0
   t_end= 2*n_cycles*np.pi
   mean_sig=2

   t_max= n_cycles*np.pi
   pred_points=100
   pred_h = ((t_end-t_start)/n) * pred_points
   clear_plt = 0
   n_test = int(pred_points)
   mu_0= 0 #prior mean
   h_0 = 1 #initializaton of scaling param
   l_0 = [1] #initializaton of time-scale param
   σ = 0.0001
   λ = 0.97
   R_th = 0.0001
   delta = np.zeros((n,n))
   for j in range(n):
       delta[j, j] = λ ** (n - j)


   #generate sine wave

   X_data = np.linspace(t_start,t_end,n)
   X_data = X_data.reshape(n, 1)
   X_data_n =X_data
   x_t=0
   X_test = np.linspace(t_start + x_t, t_end + pred_h + x_t, n_test)
   X_test = X_test.reshape(n_test, 1)
   Y_data= gen_sine(X_data,f=1, mean=mean_sig)
   Y_data_n =Y_data
   U = np.linspace(t_start,t_end,n_ind)
   U = U.reshape(n_ind, 1)




    #Implementation of AGP
    # compute new mu_λs, var_λs
   x_t = X_data[len(X_data) - 1] + 0.01
   x_test = X_test[len(X_test) - 1] + 0.1

   y_t = gen_sine(x_t, f=1, mean=mean_sig)
   U_n = U
   U_n = U
   count = 1
   dx = 0.2


   #print('delta', delta)
   U_0 = U
   #X_test = x_t + 1
   #X_test = X_test.reshape(1, 1)
   #X_test = np.linspace(t_max+ x_t, t_max + 0.1 + x_t, n_test)
   #X_test = X_test.reshape(n_test, 1)
   as_gp = Adaptive_Sparse_GPR(h_0, l_0, X_data, Y_data, U_0, X_test,σ, λ, delta,R_th)



   x_t =X_data[len(X_data)-1]
   for it in range(1000):
       x_t = x_t + dx
       #x_t = random.uniform(0, 2*np.pi)

       X_test = np.linspace(t_start, x_t + t_end + 1, n_test)
       #X_test = np.linspace(0, 2*np.pi, n_test)
       X_test = X_test.reshape(n_test, 1)
       x_test = x_test  + dx
       y_t = gen_sine(x_t ,f=1, mean=mean_sig)
       x_t = np.array([x_t])
       x_t = x_t.reshape(1, 1)
       y_t = np.array([y_t])
       y_t = y_t.reshape(1, 1)
       Y_U = np.zeros(len(U))  # calculate Y values of inducing points for visualisation
       Y_U = gen_sine(U, f=1, mean=mean_sig)

       #X_test = np.append(X_test, x_test)
       #X_test = np.delete(X_test, 0)

       #X_test = X_test.reshape(len(X_test), 1)
       #X_test = x_t +0.1
       #X_test = X_test.reshape(len(X_test), 1)
       g_t = gen_sine(X_test, f=1, mean=mean_sig)
       g_t = g_t.reshape(len(X_test), 1)
       start_time1 = time.time()
       [mu_λs, var_λs, U_gp] = as_gp.fast_adaptive_gp(y_t, x_t,X_test, it,U)
       end_time1 = time.time()
       #[mu_λs, var_λs] = as_gp.basic_gp(Y_data_n, X_data_n, X_test)
       Y_data_n = np.append(Y_data_n, y_t)
       Y_data_n = np.delete(Y_data_n, 0)
       Y_data_n = Y_data_n.reshape(len(Y_data_n), 1)

       X_data_n = np.append(X_data_n, x_t)
       X_data_n = np.delete(X_data_n, 0)
       X_data_n = X_data_n.reshape(len(X_data_n), 1)
       U = np.linspace(X_data_n[0], X_data_n[len(X_data_n)-1], n_ind)

       #U_n = np.append(U_n, x_t)
       #U_n = np.delete(U_n, 0)
       #U_n = U_n.reshape(len(U_n), 1)
       clear_plt = 1
       moving = 1
       plot_static(X_data, Y_data, X_test, mu_λs, var_λs*0.0,  U_gp, Y_U*0.0,x_t,y_t,'blue',g_t,'Online sgp',clear_plt,moving)
       clear_plt = 0


       start_time2 = time.time()
       [mu_λs, var_λs] = as_gp.basic_gp(Y_data_n, X_data_n, X_test)
       end_time2 = time.time()

       # The differnce is the elapsed time
       elapsed_time1 = end_time1 - start_time1

       elapsed_time2 = end_time2 - start_time2
       plot_static(X_data, Y_data, X_test, mu_λs, var_λs*0, U*0.0, Y_U,x_t,y_t,'green',g_t,'Dense gp',clear_plt,moving)

       #print("Elapsed time:" ,elapsed_time1," seconds")
       #[mu_λs, var_λs] = as_gp.vfe_sgp(Y_data_n, X_data_n, X_test,x_t,U)


       #plot_static(X_data_n, Y_data_n, X_test, mu_λs, var_λs*0, U*0.0, Y_U,x_t,y_t,'red',g_t, 'Sparse VFE gp',clear_plt)



   plt.show()


