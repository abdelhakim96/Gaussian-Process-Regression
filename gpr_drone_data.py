import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plot_utils import plot_gp, plot_gp_dynamic,plot_gp_animation
from get_drone_data import calculate_force_estimates_and_obtain_data
from gaussian_process_regression import Gaussian_Process_Regression
import os

"""
This is a Python script that demonstrates Gaussian Process Regression for data generated from drone simulation

The script performs various steps, including data preprocessing, model training,
and visualization of the regression results.

Author: Hakim Amer
Date: May 15, 2023
"""



if __name__ == '__main__':
    # find root directory
    gp_regression = Gaussian_Process_Regression()  # create instance of object gp_regression
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, 'figs')
    root_dir = os.path.dirname(figs_dir)

    # get data and calculate force estimates
    [gt_x, gt_y, gp_pr_x, gp_pr_y, pos_x, vel_x] = calculate_force_estimates_and_obtain_data()

    #specify parameters for regression
    sim_time = 300 # simulation time
    n_i = 200    # number of data points for initial GP
    n = 200     # number of data points for online GP
    mu_0 = 0   # prior mean
    skip = 5  # data points to skip
    dist_a = gt_x[n_i::skip]   # get the disturbance in x
    x2 = [0]                   # get the disturbance difference in x
    x2 = np.append(x2, np.array(np.diff(gt_x[n_i::skip])))

    #input data for initial GP
    x1_i = gt_x[1::skip]
    x1_i = np.array(dist_a[(1):(n_i)])

    x2_i = [0]
    x2_i = np.append(x2, np.array(np.diff(gt_x[1::skip])))
    x2_i = x2_i[0:(len(x1_i))]
    x_i = np.array([x1_i,x2_i])
    x_i = x2_i

    y_i = dist_a[(2):(n_i+1)]
    # initial value of hyperparams
    h0=1
    l0=[1, 1]
    l0= 1
    mu_all =[]
    # optimize hyper parameters of initial GP
    h = h0
    l = [l0, l0]
    #[l0, h0] = gp_regression.hyper_param_optimize(x_i, y_i,l0,h0)

    t_mu = []


   # run simulation
    for i in range (sim_time):

        x1 = np.array(dist_a[(n_i+1+i):(n_i+n+i)])
        x2 = x2[0:(len(x1))]
        x = np.array([x1,x2])


        y = dist_a[n_i+(2+i):(n_i+n+1+i)]
        x_s1= np.array([y[len(y)-1]])
        x_s2 = np.array([y[len(y) - 1]- y[len(y) - 2]])
        x_s = np.array([x_s1,x_s2])

        [mu, cov] = gp_regression.basic_gp(h, l, mu_0, y, x, x_s)
        y_all = dist_a[(3):(n+n_i+1+i)]
        mu_all = np.append(mu_all, mu)
        t = np.linspace(i+n_i, len(y)+n_i+i, len(y))
        t_all = np.linspace( 0, len(y_all)+1 , len(y_all))
        t_mu = np.append(t_mu, t_all[len(t_all) - 1] + 1)
        t_s = np.linspace(i, len(mu)+i, len(mu)+i)
        plot_gp_dynamic(y_all,y, t_all,t, t_mu,x_s, mu_all, cov,mu_all, cov,i)
