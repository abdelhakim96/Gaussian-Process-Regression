import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils import gen_sine
import random

class Adaptive_Sparse_GPR(object):
    def __init__(self, h_0, l_0, X_data, Y_data, U_0, X_test,σ, λ, Delta,R_th ):
        self.Rth= R_th
        self.h= h_0
        self.l= l_0
        self.λ = λ
        self.σ = σ
        self.delta = Delta
        self.C_t1 = None
        self.U = U_0
        self.U_n = U_0
        self.T = int(len(X_data))
        self.B_λk = None
        self.Kuu_k1= None
        self.Kux_k1 = None
        self.Kus_k1 = None
        self.KK = None
        self.k_t1 = None
        self.B_λ = None
        self.X_data = X_data
        self.Y_data = Y_data
        self.x_t= X_data[len(X_data) - 1].reshape(1, 1)
        self.u_k1= X_data[len(X_data) - 1].reshape(1, 1)

        print('AGPR Constructor')
        self.initialize_matrices_at_it_0(h_0, l_0, X_data, Y_data, X_test,Delta )

        
    def initialize_matrices_at_it_0(self, h, l, X_data, Y_data, X_test,delta ):
        #self.U = U
        self.k_t1 = self.squared_exp_kernel(h,l,self.U, X_test[len(X_test)-1])
        self.Kuu = self.squared_exp_kernel(h,l,self.U, self.U)
        self.Kux = self.squared_exp_kernel(h,l,self.U, self.U)
        self.Kus = self.squared_exp_kernel(h,l,self.U, self.U)

        self.K_us = self.squared_exp_kernel(h, l, self.U, X_test)
        self.K_ux = self.squared_exp_kernel(h, l, self.U, X_data)
        self.K_uu = self.squared_exp_kernel(h, l, self.U, self.U)
        self.k_t1 = self.squared_exp_kernel(h, l, self.U, self.x_t)
        self.K_uxk1 = self.squared_exp_kernel(h, l, self.U, self.u_k1)
        self.C = self.K_ux@ delta @ Y_data
        self.KK = self.K_ux @ delta @ np.transpose(self.K_ux)



    def squared_exp_kernel(self, h, delta, x1, x2):


        K = np.ones((len(x1), len(x2)))

        n_inputs = np.shape(x1)[1]

        delta = np.diag(delta)
        diffx = x1[:, None] - x2
        a = diffx @ np.linalg.pinv(delta)


        dist = np.zeros((len(x1), len(x2)))
        for i in range(len(x1)):
            for j in range(len(x2)):
                dist[i, j] = np.dot(diffx[i, j, :].T, np.linalg.pinv(delta) @ diffx[i, j, :])

        K = h * np.exp(-dist)

        return K


    def basic_gp(self,y, x, x_s):
        K_xx =  self.squared_exp_kernel(self.h, self.l, x, x)
        K_x_x =  self.squared_exp_kernel(self.h, self.l, x_s, x)
        K_xx_ =  self.squared_exp_kernel(self.h, self.l, x, x_s)
        K_x_x_ =  self.squared_exp_kernel(self.h, self.l, x_s, x_s)



        K_inv = np.linalg.pinv(K_xx)
        mu = K_x_x @ K_inv @ y
        cov = K_x_x_ - K_x_x @ K_inv @ K_xx_


        return mu, cov

    #def fast_adaptive_gp (self, y_t, x_t, X_data,Y_data,X_test, U_n, h,l,mu_0,σ,λ,delta,it):
    def fast_adaptive_gp(self, y_t, x_t, X_test, it):
        ###################################
        #PART1: Add new data point
        ###################################
        R_th = self.Rth
        u_k1 = x_t
        self.U = np.vstack([self.U, u_k1])
        self.U = np.delete(self.U, 0)
        self.U = self.U.reshape(len(self.U), 1)
        self.k_t1 = self.squared_exp_kernel(self.h, self.l,self.U, x_t)
        self.K_us = self.squared_exp_kernel(self.h, self.l,self.U, X_test)
        self.K_uu = self.squared_exp_kernel(self.h, self.l,self.U, self.U)


        # Calculate recursive terms when adding new data (eq. 17-18)
        self.C = self.λ *  self.C + self.k_t1 @ y_t
        self.KK = self.λ * self.KK + self.k_t1 @ np.transpose(self.k_t1)   # KK_t+1 =  λ KK_t + k_uxt * np.transpose(Kuxt))

        self.B_λ = np.linalg.inv(self.K_uu + self.σ **(-2) * (self.KK ))          #eq. 18

        # Calculate prediction mean and variance after adding new data point
        mu_λs = self.σ **(-2) * np.transpose(self.K_us) @ (self.B_λ)@ ( self.C)

        #var_λs = K_ss + K_su@ ((B_λt1)- np.linalg.pinv(K_uu))@K_us

        ###################################
        # PART2: Add new inducing point
        ###################################
        R_tot = 0
        #Check if inducing point is needed\
        for td in range(self.T):
            k_tdtd = self.squared_exp_kernel(self.h, self.l, self.X_data[td].reshape(1, 1), self.X_data[td].reshape(1, 1))
            k_tdu = self.squared_exp_kernel(self.h, self.l, self.X_data[td].reshape(1, 1), self.U)
            k_utd = self.squared_exp_kernel(self.h, self.l, self.U, self.X_data[td].reshape(1, 1))
            K_uu =  self.squared_exp_kernel(self.h, self.l, self.U, self.U)
            #R_m = λ **(T-td) * k_tdm@ np.linalg.pinv(K_mm) @k_mtd
            R_tot = R_tot + self.λ ** (self.T - td) * (k_tdtd - np.transpose(k_utd) @ np.linalg.pinv(self.K_uu) @ k_utd)

        flag = 0
        print(R_tot )
        if (R_tot > 0 and flag):


            #TODO implement this for variance calculation
            # Add new inducing point
            print('add new inducing point')


            # terms for b1 and b2 calculation (non iterative)
            k_xxt1 = self.squared_exp_kernel(self.h, self.l,self.X_data, u_k1)
            k_k1k1 = self.squared_exp_kernel(self.h, self.l,u_k1, u_k1)
            K_uxk1 = self.squared_exp_kernel(self.h, self.l,self.U, u_k1)

            b1_k1 = self.K_uxk1 + self.σ**-2 * self.K_ux @ self.delta @k_xxt1
            b2_k1 = k_k1k1 + self.σ**-2 *np.transpose(k_xxt1)@self.delta@k_xxt1

            B_A1 = np.vstack([np.linalg.pinv(self.B_λ) , np.transpose(b1_k1)] )
            B_A2 = np.vstack([b1_k1,b2_k1] )

            self.B_λ = np.linalg.pinv(np.hstack([B_A1, B_A2]))

            self.K_us = np.vstack([self.K_us, self.squared_exp_kernel(self.h, self.l, u_k1, X_test)])
            self.K_ux = np.vstack([self.K_ux, self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data)])

            # TODO: calculate more effeceintly using properties of block inversion

            self.K_uxk1 = np.vstack([self.K_uxk1, self.squared_exp_kernel(self.h, self.l, u_k1, u_k1)])

            self.C = np.vstack([self.C,  self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data)
                                   @ self.delta @ self.Y_data])

            a1= self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data) @ \
                self.delta @ self.squared_exp_kernel(self.h, self.l, self.X_data, self.U)
            a2 = self.squared_exp_kernel(self.h, self.l, self.U, self.X_data) @ \
                 self.delta @ self.squared_exp_kernel(self.h, self.l, self.X_data, u_k1)
            a3 = self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data) @ \
                 self.delta @ self.squared_exp_kernel(self.h, self.l, self.X_data, u_k1)

            A = np.vstack([self.KK,a1])
            B = np.vstack([a2,a3])
            self.KK = np.hstack([A,B])

            au1 = self.squared_exp_kernel(self.h, self.l, u_k1, self.U)
            au2 = self.squared_exp_kernel(self.h, self.l, self.U, u_k1)
            au3 = self.squared_exp_kernel(self.h, self.l, u_k1, u_k1)

            Au = np.vstack([self.K_uu, au1])
            Bu = np.vstack([au2, au3])
            self.K_uu = np.hstack([Au, Bu])
            #print('l U', len(self.U))
            self.k_t1 =  np.vstack([self.k_t1, self.squared_exp_kernel(self.h, self.l, u_k1, x_t)])
            mu_λs = self.σ ** (-2) * np.transpose(self.K_us) @ self.B_λ @ self.C
            self.U = np.vstack([self.U, u_k1])
            #self.U = np.delete(self.U, 0)
            self.U = self.U.reshape(len(self.U), 1)
            #     var_λs_k1 = K_ss + K_su_k1 @ ((B_λk1) - np.linalg.pinv(K_uu_k1)) @ K_us_k1


        return mu_λs, 0.0