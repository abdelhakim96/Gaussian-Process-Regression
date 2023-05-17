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
    n_data = 80
    n_ind = 10
    n_test = 100
    period = 1
    amplitude = 1

    #Generate Signal
    x1 = np.linspace(0, 2 * np.pi, n_data)

    #x = x.reshape(-1, 1)

    x1_s = np.linspace(0, 2 * np.pi, n_test)

    #x_s = x_s.reshape(-1, 1)

    [y1,u,xn,yn]= gp_regression.generate_sine(period, amplitude,x1,n_data, n_ind, n_test)
    [ys,u,xx,yn]= gp_regression.generate_sine(period, amplitude,x1_s,n_test, n_test, n_test)
    x2_s = np.diff(ys)
    x2_s = np.insert(x2_s, 0, 0)
    x_s = np.c_[x1_s, x2_s]

    x_s = x1_s.reshape(-1,1)
    x_s = x1_s
    x2 = np.diff(y1)/np.diff(x1)
    x2 = np.insert(x2, 0, 0)
    x = np.c_[x1, x2]
    x=x1
   # x = x1.reshape(-1,1)
    y = np.c_[y1, y1]
    y=y1
    #y = y1.reshape(-1,1)
    #y = y.reshape(-1, 1)
    step_size =  (x1[2] - x1[1])


    #y = y.reshape(-1, 1)
    #step_size = 1
    #Optimize hyperparameters of the GP
    h=1

    #[l, h]= gp_regression.hyper_param_optimize( x, y)
    [h, l] =gp_regression.hyper_param_opt(x, y, 0)
    #l= np.diag(l)

    print(l)
    print(h)

    #Compute and Vizualize
    sim_time =1
    frames = []

    for i in range (sim_time):




        [mu, cov] = gp_regression.basic_gp(h, l, mu_0, y, x, x_s)
        #[mu, cov] = offline_sparse_gp_FITC(h, l, mu_0, y, x, x_s, u)


        it=i
        gif_path = os.path.join(figs_dir, 'animation.gif')

        #y_train = y[:,0]
        y_train = y
        x_train = x
        x_test = x_s
        #t_train = x_train[:,0]
        t_train = x_train
        #t_test = x_test[:, 0]
        t_test = x_test
        #mu = mu[:,0]
        plot_gp_dynamic(y_train, x_train, x_test, t_train,t_test, mu, cov, pred_ahead)

       # plot_gp_dynamic(y1, y1, x1, x1, x1_s, x1_s, mu, cov, pred_ahead)
        x = np.append(x, x[-1] + step_size)
        off_u = 0.01 #offset to prevent x=u
        u = np.append(u, u[-1] + step_size - off_u )

        u = u[1:]
        x = x[1:]
        x_s = x_s[1:]
        y = gp_regression.generate_sine(period, amplitude,x,n_data, n_ind, n_test)[0]

    #Plotting
    #plot_gp(y,x,x_s,mu,cov, mu_s, cov_s)