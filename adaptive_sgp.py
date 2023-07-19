import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils import gen_sine

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



        for i in range(len(X_data)):
            delta[i, i] = λ ** (len(X_data) - i)




        T = int(len(X_data))
        R_th = 0.0001
        K_xx = self.squared_exp_kernel(h, l, X_data, X_data)
        K_uu = self.squared_exp_kernel(h, l, U, U)
        K_ux = self.squared_exp_kernel(h, l, U, X_data)
        K_xu = self.squared_exp_kernel(h, l,X_data, U)
        K_ss = self.squared_exp_kernel(h, l,X_test, X_test)
        K_su = self.squared_exp_kernel(h, l,X_test, U)
        K_us = self.squared_exp_kernel(h, l,U, X_test)

        #note: check if K_xu = K_ux^T
        u_k1 = np.array([1])
        u_k1 = u_k1.reshape(1, 1)
        k_t1 = np.dot(U,x_t)
        #k_t_T = np.dot(U,X_data[])


        B_λ= np.linalg.pinv(K_uu + σ **(-2) * (λ * (K_ux @ delta @ K_xu) +
                                               np.transpose(k_t1) @ k_t1) )

        mu_λs = σ **(-2) * K_su @ (B_λ)@ ( λ * (K_ux @ delta @Y_data) + np.transpose(k_t1)*y_t)
        var_λs = K_ss + K_su@ ((B_λ)- np.linalg.pinv(K_uu))@K_us


        #Add new inducing point

        #Check if inducing point is needed\
        for td in range(T):
            k_tdtd = self.squared_exp_kernel(h, l, X_data[td].reshape(1, 1), X_data[td].reshape(1, 1))
            k_tdu = self.squared_exp_kernel(h, l, X_data[td].reshape(1, 1), U)
            k_utd = self.squared_exp_kernel(h, l, U, X_data[td].reshape(1, 1))
            #R_m = λ **(T-td) * k_tdm@ np.linalg.pinv(K_mm) @k_mtd
            R_tot = λ ** (T - td) * (k_tdtd - k_tdu @ np.linalg.pinv(K_uu) @ k_utd)
       # M = len(U)
       # for m in M:
       #     R_tot = R_m


        if (R_tot>0):
            u_k1= x_t.reshape(1, 1)
            # Add new inducing point



            A = np.transpose(self.squared_exp_kernel(h, l,X_test, u_k1))
            K_su_k1 = np.hstack([K_su, self.squared_exp_kernel(h, l,X_test, u_k1)])
            K_ux_k1 = np.hstack([K_ux, self.squared_exp_kernel(h, l,U, u_k1)])
            K_xx_t1 = np.hstack([K_xx, self.squared_exp_kernel(h, l,X_data, u_k1)])



            # Add to K_uu
            K_uu_k1 =  np.hstack([K_uu, self.squared_exp_kernel(h, l,U, u_k1)])
            K_uu_k1 = np.hstack([K_uu, self.squared_exp_kernel(h, l, U, u_k1)])
            a = np.hstack([self.squared_exp_kernel(h, l, u_k1, U), self.squared_exp_kernel(h, l, u_k1, u_k1)])
            K_uu_k1 = np.vstack([K_uu_k1, a])

            #calculate new B_λ_k1


            k_xt1x = self.squared_exp_kernel(h, l,u_k1, X_data)
            k_xxt1 = self.squared_exp_kernel(h, l,X_data, u_k1)
            k_k1k1 = self.squared_exp_kernel(h, l,u_k1, u_k1)
            K_uxk1 = self.squared_exp_kernel(h, l,U, u_k1)
            b1_k1 = K_uxk1 + σ**-2 *K_ux@delta@k_xxt1
            b2_k1 = k_k1k1 + σ**-2 *k_xt1x@delta@k_xxt1
            B_A1 = np.vstack([np.linalg.pinv(B_λ),np.transpose(b1_k1)] )
            B_A2 = np.vstack([b1_k1,b2_k1] )

            B_λk1 = np.hstack([B_A1, B_A2])

            X_data = np.vstack([X_data,u_k1])
            Y_data = np.vstack([Y_data, gen_sine(X_data,f=1, mean=0)])
            U = np.vstack([U,u_k1])

            delta = np.zeros((len(X_data), len(X_data)))
            for i in range(len(X_data)):
                delta[i, i] = λ ** (len(X_data) - i)


            mu_λs_k1 = σ ** (-2) * K_su_k1 @ B_λk1 @ (K_ux_k1 @ delta @ Y_data)
            var_λs_k1 = K_ss + K_su_k1 @ ((B_λk1) - np.linalg.pinv(K_uu_k1)) @ K_su_k1



        return  mu_λs_k1, var_λs_k1