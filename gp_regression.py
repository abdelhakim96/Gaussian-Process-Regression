import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plot_utils import plot_gp


def offline_sparse_gp_FITC(h,l,mu_0, y, x, xs,u):
    # Create covariance matrices

    Kff =  squared_exp_fun(h, l, x, x)
    Kfu =  squared_exp_fun(h, l, x, u)
    Kuu =  squared_exp_fun(h, l, u, u)
    Kuf =  squared_exp_fun(h, l, u, x)
    Kus =  squared_exp_fun(h, l, u, xs)
    Ksu = squared_exp_fun(h, l, xs, u)
    Kss = squared_exp_fun(h, l, xs, xs)



    Qff = Kfu @ np.linalg.inv(Kuu) @ Kuf
    delta_ff = np.diag((Kff - Qff)[0])
    m_u = 0
    m_s = 0



    cov_u = Kuu @ np.linalg.inv(Kuu + Kuf @ np.linalg.inv(delta_ff)@ Kfu) @ Kuu
    mu_u = m_u + cov_u @ np.linalg.inv(Kuu) @ Kuf @ np.linalg.inv(delta_ff) @ y

    mu_s = m_s + Ksu @  np.linalg.inv(Kuu) @ mu_u
    cov_s = Kss - Ksu @ np.linalg.inv(Kuu) @ (Kuu - cov_u) @ np.linalg.inv(Kuu) @ Kus
    return mu_s, cov_s



def online_sparse_gp_FITC(x, xs,xn,yn, y,h,l,u):
    # Create covariance matrices
    Kff =  squared_exp_fun(h, l, x, x)
    Kfu =  squared_exp_fun(h, l, x, u)
    Kuu =  squared_exp_fun(h, l, u, u)
    Kuf =  squared_exp_fun(h, l, u, x)
    Kus =  squared_exp_fun(h, l, u, xs)
    Ksu = squared_exp_fun(h, l, xs, u)
    Kss = squared_exp_fun(h, l, xs, xs)

    Kun =  squared_exp_fun(h, l, u, xn)    # matrix for data at latest time, t=n
    Knu = squared_exp_fun(h, l, xn, u)
    Knn = squared_exp_fun(h, l, xn, xn)

    n= len(Kuu)


    Qff = Kfu @ np.linalg.inv(Kuu) @ Kuf
    Qnn = Knu @ np.linalg.inv(Kuu) @ Kun

    delta_ff = np.diag((Kff - Qff)[0])

    delta_nn = np.diag((Knn - Qnn)[0])
    m_u = 0
    m_s = 0



    cov_u = Kuu @ np.linalg.inv(Kuu + Kuf @ np.linalg.inv(delta_ff)@ Kfu) @ Kuu
    mu_u = m_u + cov_u @ np.linalg.inv(Kuu) @ Kuf @ np.linalg.inv(delta_ff) @ y

    #calculate at current time n

    P_n = np.linalg.inv(Kuu) @ Kun @ np.linalg.inv(delta_nn) @ Knu @ np.linalg.inv(Kuu)
    cov_u_n = cov_u - (cov_u @ P_n @ cov_u)/(1+np.trace(cov_u @ P_n))

    a = (np.eye(n) - (cov_u_n@ P_n)/(1+np.trace(cov_u@P_n)))@ mu_u
    mu_u_n = (np.eye(n) - (cov_u_n@ P_n)/(1+np.trace(cov_u@P_n)))@ mu_u + cov_u_n@ np.linalg.inv(Kuu) @ Kun@delta_nn@yn

    mu_s = m_s + Ksu @  np.linalg.inv(Kuu) @ mu_u_n
    cov_s = Kss - Ksu @ np.linalg.inv(Kuu) @ (Kuu - cov_u_n) @ np.linalg.inv(Kuu) @ Kus
    return mu_s, cov_s







def log_max_likelyhood(x, y, noise):
    def step(theta):
        K = squared_exp_fun(theta[1], theta[0], x, x)
        return np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
               0.5 * y.T @ np.linalg.pinv(K) @ y + \
               0.5 * len(x) * np.log(2*np.pi)
    return step

# minimize -log liklihood

def hyper_param_optimize(x, y):
    res = minimize(log_max_likelyhood(x, y, 0.0), [1, 1],
               bounds=((1e-20, None), (1e-20, None)),
               method='L-BFGS-B')

    l_opt, sigma_f_opt = res.x
    return l_opt, sigma_f_opt



def squared_exp_fun(h, l, x1, x2):
    K = (h ** 2)  * np.exp((-(x1[:, None] - x2) ** 2 )/ (2 * l ** 2))

    return K




def basic_gp(h,l,mu_0, y, x, x_s):
    K_xx = squared_exp_fun(h, l, x, x)
    K_x_x = squared_exp_fun(h, l, x_s, x)
    K_xx_ = squared_exp_fun(h, l, x, x_s)
    K_x_x_ = squared_exp_fun(h, l, x_s, x_s)

    K_inv = np.linalg.inv(K_xx)
    mu = mu_0 + K_x_x @ K_inv @ y
    cov = K_x_x_ - K_x_x @ K_inv @ K_xx_


    return mu, cov

if __name__ == '__main__':
    # Step 1: Define GP parameters
    mu_0 = 0.0 #prior mean
    h = 1 #amplitude coff
    l = 1 #timescale

    #optimization params
    learning_r = 0.0001 #for gradient descent
    iter = 2000

    # Step 2: Generate data for signal (sine wave)
    n_data = 20
    n_ind = 10
    n_test = 100
    period = 1
    x = np.linspace(0, 2 * np.pi, n_data)
    y = np.sin(period  * x)
    x_s = np.linspace(0, 2 * np.pi, n_test )
    u = np.linspace(0, 2 * np.pi, n_ind)


    #Optimize hyperparameters of the GP
    [l, h] = hyper_param_optimize(x, y)


    #Select GP Type
    [mu,cov] = basic_gp(h,l,mu_0, y, x, x_s)
    [mu_s, cov_s] = offline_sparse_gp_FITC(h,l,mu_0, y, x, x_s,u)

    xn = np.array([x[len(x)-1]+0.001])
    yn = np.array([y[len(y) - 1] + 0.001])
    [mu_s_on, cov_s_on] = online_sparse_gp_FITC(x, x_s,xn,yn, y,h,l,u)
    print(len(mu_s_on))
    print(len(mu_s))
    #Plot
    #plot_gp(y,x,x_s,mu,cov, mu_s, cov_s)
    plot_gp(y, x, x_s, mu, cov, mu_s_on, cov_s_on)
    #plot_gp(y,x,x_s,mu,cov, mu, cov)





