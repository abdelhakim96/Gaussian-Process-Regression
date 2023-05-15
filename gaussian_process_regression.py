import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Gaussian_Process_Regression(object):
    #def __init__(self):
       # print('GPR Constructor')

    def offline_sparse_gp_FITC(self,h,l,mu_0, y, x, xs,u):
        # Create covariance matrices

        Kff =  self.squared_exp_fun(h, l, x, x)
        Kfu =  self.squared_exp_fun(h, l, x, u)
        Kuu =   self.squared_exp_fun(h, l, u, u)
        Kuf =   self.squared_exp_fun(h, l, u, x)
        Kus =   self.squared_exp_fun(h, l, u, xs)
        Ksu =  self.squared_exp_fun(h, l, xs, u)
        Kss =  self.squared_exp_fun(h, l, xs, xs)

        Qff = Kfu @ np.linalg.inv(Kuu) @ Kuf
        delta_ff = np.diag((Kff - Qff)[0])
        m_u = 0
        m_s = 0

        cov_u = Kuu @ np.linalg.inv(Kuu + Kuf @ np.linalg.pinv(delta_ff)@ Kfu) @ Kuu
        mu_u = m_u + cov_u @ np.linalg.pinv(Kuu) @ Kuf @ np.linalg.pinv(delta_ff) @ y

        mu_s = m_s + Ksu @  np.linalg.pinv(Kuu) @ mu_u
        cov_s = Kss - Ksu @ np.linalg.pinv(Kuu) @ (Kuu - cov_u) @ np.linalg.pinv(Kuu) @ Kus
        return mu_s, cov_s



    def online_sparse_gp_FITC(self,x, xs,xn,yn, y,h,l,u):
        # Create covariance matrices
        Kff =   self.squared_exp_fun(h, l, x, x)
        Kfu =   self.squared_exp_fun(h, l, x, u)
        Kuu =   self.squared_exp_fun(h, l, u, u)
        Kuf =   self.squared_exp_fun(h, l, u, x)
        Kus =   self.squared_exp_fun(h, l, u, xs)
        Ksu = self.squared_exp_fun(h, l, xs, u)
        Kss =  self.squared_exp_fun(h, l, xs, xs)

        Kun =   self.squared_exp_fun(h, l, u, xn)    # matrix for data at latest time, t=n
        Knu =  self.squared_exp_fun(h, l, xn, u)
        Knn =  self.squared_exp_fun(h, l, xn, xn)

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

    def log_max_likelyhood(self,x, y, noise):
        def step(theta):
            K =  self.squared_exp_fun(theta[1], theta[0], x, x)

            return np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
                   0.5 * y.T @ np.linalg.pinv(K) @ y + \
                   0.5 * len(x) * np.log(2*np.pi)
        return step

# minimize -log liklihood

    def hyper_param_optimize(self,x, y):
        res = minimize( self.log_max_likelyhood(x, y, 0.0), [1, 1],
                   bounds=((1e-20, None), (1e-20, None)),
                   method='L-BFGS-B')

        l_opt, sigma_f_opt = res.x
        return l_opt, sigma_f_opt



    def squared_exp_fun(self,h, l, x1, x2):
        #K = (h ** 2)  * np.exp((-(x1[:, None] - x2) ** 2 )/ (2 * l ** 2))

        if (x1.ndim > 1) or (x2.ndim > 1):
            K = np.zeros([np.shape(x1)[1], np.shape(x1)[1]])

            for i in range(np.shape(x1)[0]):
                K = K + (h ** 2) * (np.exp((-(x2[i, :] - np.tile(x1[i, :], (x2.shape[1], 1)).T) ** 2) / (2 * l[i] ** 2)))

            #K = np.sum((h ** 2) * np.exp((-(x2[:, :] - np.tile(x1[:, :], (x2.shape[1], 1)).T) ** 2) / (2 * l ** 2)), axis=1)
        else:
            K = (h ** 2)  * np.exp((-(x1[:, None] - x2) ** 2 )/ (2 * l ** 2))


        return K

    def multi_input_squared_exp_fun(h, delta, x1, x2):
        #K = (h ** 2)  * np.exp((-0.5 *  (x1 - x2).T @ delta @ (x1 - x2) ))
        squared_dist = np.sum((x1.T[:, None] - x2) ** 2, axis=-1)
        K = (h ** 2) * np.exp(-squared_dist / (2 * delta ** 2))

        return K

    def generate_sine(self,period,amplitude,x,n_data, n_ind,n_test):
        if x == []:
            x = np.linspace(0, 2 * np.pi, n_data)

        x_s = np.linspace(0, x[len(x)-1], n_test)
        u = np.linspace(0, 2 * np.pi, n_ind)
        xn = np.array([x[len(x)-1]+0.001])
        y = amplitude * np.sin(period  * x)
        yn = np.array([y[len(y) - 1] + 0.001])

        return [y,x, x_s,u,xn,yn]

    def example_system(x0, n):

        x = np.array(np.zeros(n))
        x[0] = x0
        for i  in range(len(x)-1):
            x[i+1] = x[i]/2 + (25 * x[i] / (1 + (x[i]) ** 2)) * np.cos(x[i]) + np.random.normal(0,1)


        return x


    def basic_gp_2(self,h,l,mu_0, y, x, x_s):



        K_xx = self.multi_input_squared_exp_fun(h, l, x, x)
        K_x_x =  self.multi_input_squared_exp_fun(h, l, x_s, x)
        K_xx_ =  self.multi_input_squared_exp_fun(h, l, x, x_s)
        K_x_x_ =  self.multi_input_squared_exp_fun(h, l, x_s, x_s)

        K_inv = np.linalg.inv(K_xx)
        mu = mu_0 + K_x_x @ K_inv @ y
        cov = K_x_x_ - K_x_x @ K_inv @ K_xx_


        return mu, cov
    def basic_gp(self,h,l,mu_0, y, x, x_s):
        K_xx =  self.squared_exp_fun(h, l, x, x)
        K_x_x =  self.squared_exp_fun(h, l, x_s, x)
        K_xx_ =  self.squared_exp_fun(h, l, x, x_s)
        K_x_x_ =  self.squared_exp_fun(h, l, x_s, x_s)



        K_inv = np.linalg.pinv(K_xx)
        mu = K_x_x @ K_inv @ y
        cov = K_x_x_ - K_x_x @ K_inv @ K_xx_


        return mu, cov