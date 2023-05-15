import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plot_utils import plot_gp, plot_gp_animation, plot_gp_dynamic
import os
from gaussian_process_regression import Gaussian_Process_Regression


"""
This is a Python script that demonstrates Gaussian Process Regression example for a simple generated sine wave.

The script performs various steps, including data preprocessing, model training,
and visualization of the regression results.

Author: Hakim Amer
Date: May 15, 2023
"""






if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(script_dir, 'figs')
    root_dir = os.path.dirname(figs_dir)
    gp_regression = Gaussian_Process_Regression()  # create instance of object gp_regression

    # Step 1: Define GP parameters
    mu_0 = 0.0 #prior mean
    h = 1 #amplitude coff
    l = 1 #timescale

    #simulation params
    animate = 1 #Flag to determine if you want to animate TRUE: will create animation, FALSE: will just plot
    pred_ahead = 2
    sim_time = 10

    #optimization params
    learning_r = 0.0001 #for gradient descent
    iter = 2000

    # Input Signal params
    n_data = 20
    n_ind = 10
    n_test = 100
    period = 1
    amplitude = 1

    #Generate Signal

    [y,x, x_s,u,xn,yn]= gp_regression.generate_sine(period, amplitude,[],n_data, n_ind, n_test)
    step_size =  (x[2] - x[1])

    #Optimize hyperparameters of the GP
    [l, h] = gp_regression.hyper_param_optimize(x, y)

    #Compute and Vizualize
    sim_time =20
    frames = []
    for i in range (sim_time):

        x_s = np.linspace(0, x[len(x)-1]+pred_ahead, n_test )

        [mu, cov] = gp_regression.basic_gp(h, l, mu_0, y, x, x_s)
        #[mu, cov] = offline_sparse_gp_FITC(h, l, mu_0, y, x, x_s, u)


        it=i
        gif_path = os.path.join(figs_dir, 'animation.gif')
        plot_gp_animation(y, x, x_s, mu, cov, mu, cov, gif_path, sim_time,it,frames)

        x = np.append(x, x[-1] + step_size)
        off_u = 0.01 #offset to prevent x=u
        u = np.append(u, u[-1] + step_size - off_u )

        u = u[1:]
        x = x[1:]
        x_s = x_s[1:]
        y = gp_regression.generate_sine(period, amplitude,x,n_data, n_ind, n_test)[0]

    #Plotting
    #plot_gp(y,x,x_s,mu,cov, mu_s, cov_s)




