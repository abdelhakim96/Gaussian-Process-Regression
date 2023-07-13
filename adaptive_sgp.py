import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Adaptive_Sparse_GPR(object):
    #def __init__(self):
       # print('GPR Constructor')

    def squared_exp_kernel(self, h, delta, x1, x2):



        # multivariate
        K1 = np.ones((len(x1), len(x2)))
        K = np.ones((len(x1), len(x2)))

        n_inputs = np.shape(x1)[1]



        delta = np.diag(delta)
        # delta[1,1]=0.0
        diffx = x1[:, None] - x2
        a = diffx @ np.linalg.pinv(delta)


        dist = np.zeros((len(x1), len(x2)))
        for i in range(len(x1)):
            for j in range(len(x2)):
                dist[i, j] = np.dot(diffx[i, j, :].T, np.linalg.pinv(delta) @ diffx[i, j, :])

        K = h * np.exp(-dist)
        a = 1

        sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1)
        delta_inv_sq = np.linalg.pinv(delta) ** 2
        delta_inv_sq = np.linalg.pinv(delta) ** 2
        sqdist = sqdist - 2 * np.dot(x1, delta_inv_sq @ x2.T)

        KK = (h ** 2) * np.exp((-(x1[:, None] - x2) ** 2) / (2 * delta[0] ** 2))
        return K


    def basic_gp(self,h,l,mu_0, y, x, x_s):
        K_xx =  self.squared_exp_kernel(h, l, x, x)
        K_x_x =  self.squared_exp_kernel(h, l, x_s, x)
        K_xx_ =  self.squared_exp_kernel(h, l, x, x_s)
        K_x_x_ =  self.squared_exp_kernel(h, l, x_s, x_s)



        K_inv = np.linalg.pinv(K_xx)
        mu = K_x_x @ K_inv @ y
        cov = K_x_x_ - K_x_x @ K_inv @ K_xx_


        return mu, cov

    def fast_adaptive_gp (self, y_t, x_t, X_data,Y_data,X_test, U, h,l,mu_0,σ,λ,delta):

        K_uu = self.squared_exp_kernel(h, l, U, U)
        K_ux = self.squared_exp_kernel(h, l, U, X_data)
        K_xu = self.squared_exp_kernel(h, l,X_data, U)
        K_ss = self.squared_exp_kernel(h, l,X_test, X_test)
        K_su = self.squared_exp_kernel(h, l,X_test, U)
        K_us = self.squared_exp_kernel(h, l,U, X_test)

        #note: check if K_xu = K_ux^T

        k_t = np.dot(U,x_t)
        B_λ= np.linalg.pin(K_uu + σ^-2 (λ * (K_ux@delta@K_xu) + np.transpose(k_t) @ k_t))
        mu_λs = σ^-2 * K_su @ (B_λ)@ ( λ (K_ux@delta @Y_data) + np.transpose(k_t)*y_t)
        var_λs = K_ss + K_su ((B_λ)- np.linalg.pinv(K_uu))@K_us

        return  mu_λs, var_λs