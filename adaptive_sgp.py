import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils import gen_sine
import random

class Adaptive_Sparse_GPR(object):
    def __init__(self, h_0, l_0, X_data, Y_data, U_0, X_test,σ, λ, Delta,R_th ):
        self.flag = 0
        self.remove_flag = 0
        self.add_ind = 0
        self.add_data=0
        self.Rth= R_th
        self.h= h_0
        self.l= l_0
        self.λ = λ
        self.σ = σ
        self.R_tot = 0.0
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
        self.KK_1 = None
        self.k_t1 = None
        self.B_λ = None
        self.B_λi = None
        self.X_data = X_data
        self.Y_data = Y_data
        self.x_t= X_data[len(X_data) - 1].reshape(1, 1)
        self.u_k1= X_data[len(X_data) - 1].reshape(1, 1)
        self.mu_λs = None
        self.var_λs = None
        print('AGPR Constructor')
        self.initialize_matrices_at_it_0(h_0, l_0, X_data, Y_data, X_test,Delta )

        
    def initialize_matrices_at_it_0(self, h, l, X_data, Y_data, X_test,delta ):
        #self.U = U
        self.k_t1 = self.squared_exp_kernel(h,l,self.U, X_test[len(X_test)-1])
        self.K_us = self.squared_exp_kernel(h, l, self.U, X_test)
        self.K_ux = self.squared_exp_kernel(h, l, self.U, X_data)
        self.K_uu = self.squared_exp_kernel(h, l, self.U, self.U)
        self.k_t1 = self.squared_exp_kernel(h, l, self.U, self.x_t)
        self.C = self.K_ux@ delta @ Y_data
        self.KK = self.K_ux @ delta @ np.transpose(self.K_ux)
        self.KKi = self.K_ux @ delta @ np.transpose(self.K_ux)
        self.K_us = self.squared_exp_kernel(self.h, self.l, self.U, X_test)
        self.K_ss = self.squared_exp_kernel(self.h, self.l, X_test, X_test)
        self.K_su = self.squared_exp_kernel(self.h, self.l, X_test, self.U)

        self.B_λi = self.K_uu + self.σ ** (-2) * (self.KK)
        self.B_λ = np.linalg.pinv(self.B_λi)
        self.B_λ0 = self.K_uu + self.σ ** (-2) * (self.KK)
        self.mu_λs = self.σ ** (-2) * np.transpose(self.K_us) @ (self.B_λ) @ (self.C)

        self.var_λs = self.K_ss + self.K_su @ ((self.B_λ) - np.linalg.pinv(self.K_uu)) @ self.K_us
        #self.mu_λs = self.σ ** (-2) * np.transpose(self.K_us) @ (self.B_λ) @ (self.C)
        #self.var_λs = self.K_ss + self.K_su@ ((self.B_λ)- np.linalg.pinv(self.K_uu))@self.K_us

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

    def vfe_sgp(self,y, x, x_s,u_k1,U):


        K_us = self.squared_exp_kernel(self.h, self.l, U, x_s)
        K_uu = self.squared_exp_kernel(self.h, self.l, U, U)
        K_ux = self.squared_exp_kernel(self.h, self.l,U, x)
        K_xu = self.squared_exp_kernel(self.h, self.l,x,U)


        K_ss = self.squared_exp_kernel(self.h, self.l, x_s, x_s)

        B_λ = np.linalg.inv(K_uu + self.σ ** (-2) * (K_ux @self.delta@K_xu ))

        mu = self.σ**-2 * np.transpose(K_us)@ B_λ @ K_ux@self.delta @ y
        cov = K_ss + np.transpose(K_us) @ (B_λ - np.linalg.inv(K_uu))@K_us


        #self.U = np.linspace(self.U[0, 0], u_k1, 20)
        #self.U = self.U.reshape(20, 1)
        #self.U = np.vstack([self.U, u_k1])

        print(U)
        return mu, cov

    #def fast_adaptive_gp (self, y_t, x_t, X_data,Y_data,X_test, U_n, h,l,mu_0,σ,λ,delta,it):
    def fast_adaptive_gp(self, y_t, x_t, X_test, it,U):
        ###################################
        #PART1: Add new data point
        ###################################

        #self.U= U
        R_th = self.Rth
        u_k1 = x_t +1

        self.flag = 0
        self.k_t1 = self.squared_exp_kernel(self.h, self.l,self.U, x_t)
        self.K_us = self.squared_exp_kernel(self.h, self.l,self.U, X_test)
        self.K_su = self.squared_exp_kernel(self.h, self.l,X_test,self.U)
        self.K_ss = self.squared_exp_kernel(self.h, self.l,X_test, X_test)
        #self.K_uu = self.squared_exp_kernel(self.h, self.l, self.U, self.U)
        #self.K_ux = self.squared_exp_kernel(self.h, self.l, self.U, self.X_data)
        #self.K_xu = self.squared_exp_kernel(self.h, self.l, self.X_data, self.U)


        self.add_data=1
        if self.add_data==1:

           self.B_λ0= self.K_uu + self.σ ** (-2) * (self.KK)
        # Calculate recursive terms when adding new data (eq. 17-18)
           self.C = self.λ *  self.C + self.k_t1 @ y_t
           #self.C = self.λ *self.K_ux @ self.delta @ self.Y_data + self.k_t1 @ y_t

           self.KK = self.λ * self.KK + self.k_t1 @ np.transpose(self.k_t1)   # KK_t+1 =  λ KK_t + k_uxt * np.transpose(Kuxt))
          # self.KK = self.λ * self.K_ux @ self.delta @ np.transpose(self.K_ux)+ self.k_t1 @ np.transpose(self.k_t1)

           print('updating recursive terms in online manner')

           self.B_λi = self.K_uu + self.σ ** (-2) * (self.KK)
           #self.B_λi = self.K_uu + self.σ ** (-2) * (self.KK)

           self.B_λ = np.linalg.inv(self.B_λi)          #eq. 18

        # Calculate prediction mean and variance after adding new data point
           self.mu_λs = self.σ **(-2) * np.transpose(self.K_us) @ (self.B_λ) @ ( self.C)

           self.var_λs = self.K_ss + self.K_su@ ((self.B_λ)- np.linalg.pinv(self.K_uu))@self.K_us

        #self.C = np.vstack([self.C,  self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data)
        #                    @ self.delta @ self.Y_data])

        #a1 = self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data) @ \
        #     self.delta @ self.squared_exp_kernel(self.h, self.l, self.X_data, self.U)

        #a3 = self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data) @ \
        #     self.delta @ self.squared_exp_kernel(self.h, self.l, self.X_data, u_k1)

        #A = np.vstack([self.KK, a1])
        #B = np.vstack([np.transpose(a1), a3])

        #self.KK = np.hstack([A, B])
        #self.U = np.vstack([self.U, u_k1])

        ###################################
        # PART2: Add new inducing point
        ###################################
        self.add_ind = 0
        self.remove_flag = 0
        if len(self.U)>40:
           self.remove_flag = 1

        if (it%1==0):
            self.add_ind = 1

        #self.add_ind = 0
        #Check if inducing point is needed
        for td in range(self.T):
            k_tdtd = self.squared_exp_kernel(self.h, self.l, self.X_data[td].reshape(1, 1), self.X_data[td].reshape(1, 1))
            k_utd = self.squared_exp_kernel(self.h, self.l, self.U, self.X_data[td].reshape(1, 1))
            #R_m = self.λ **(T-td) * k_tdm@ np.linalg.pinv(K_mm) @k_mtd
            #self.R_tot = self.R_tot + self.λ ** (self.T - td) * (k_tdtd - np.transpose(k_utd) @ np.linalg.pinv(self.K_uu) @ k_utd)

        #self.U = np.vstack([self.U, u_k1])
        #self.U = np.delete(self.U, 0, axis=0)
        #self.U = self.U.reshape(len(self.U), 1)

        print(len(self.U))
        if (self.add_ind):

            #TODO implement this for variance calculation
            # Add new inducing point
            print('add new inducing point')

            #self.KK = self.λ * self.KK + self.k_t1 @ np.transpose(self.k_t1)

            au1 = self.squared_exp_kernel(self.h, self.l, u_k1, self.U)
            au3 = self.squared_exp_kernel(self.h, self.l, u_k1, u_k1)
            Au = np.vstack([self.K_uu, au1])
            Bu = np.vstack([np.transpose(au1), au3])

            # terms for b1 and b2 calculation (non iterative) #Basic Implementation
            k_xu_k1 = self.squared_exp_kernel(self.h, self.l,self.X_data, u_k1)
            k_k1_k1 = self.squared_exp_kernel(self.h, self.l,u_k1, u_k1)
            K_uu_k1 = self.squared_exp_kernel(self.h, self.l,self.U, u_k1)

            b1_k1 = K_uu_k1 + self.σ**-2 * self.K_ux @ self.delta @k_xu_k1
            b2_k1 = k_k1_k1 + self.σ**-2 * np.transpose(k_xu_k1) @self.delta @ k_xu_k1
            #b1_k1 = b1_k1 + self.σ ** (-2) * self.squared_exp_kernel(self.h, self.l, self.U, u_k1)
            b2_k1 = b2_k1 +  self.σ ** (-2) * self.squared_exp_kernel(self.h, self.l, u_k1, u_k1)
            #self.B_λi = self.B_λi + self.σ ** (-2)* self.k_t1 @ np.transpose(self.k_t1)

            B_A1 = np.vstack([self.B_λi, np.transpose(b1_k1)] )
            B_A2 = np.vstack([b1_k1,b2_k1] )
            self.B_λi = np.hstack([B_A1, B_A2])
            self.B_λ = np.linalg.inv(self.B_λi)



 #Testing stuff
           # self.B_λ0 =  self.K_uu + self.σ ** (-2) * (self.KK)

            #self.B_λ0 = self.K_uu + self.σ ** (-2) * (self.KK + self.k_t1 @ np.transpose(self.k_t1))

            #self.k_t1 = np.vstack([self.k_t1, self.squared_exp_kernel(self.h, self.l, u_k1, u_k1)] )
            #self.B_λi = self.B_λi + self.σ ** (-2) * self.k_t1 @ np.transpose(self.k_t1)

            #if self.add_data==1:
            #   self.B_λi = self.B_λi + self.σ ** (-2) * self.k_t1 @ np.transpose(self.k_t1)
            #   b1_k1 = b1_k1 + self.σ ** (-2) * self.squared_exp_kernel(self.h, self.l, self.U, u_k1)
            #   b2_k1 = b2_k1 + self.σ ** (-2) * self.squared_exp_kernel(self.h, self.l, u_k1, u_k1)
            #print('size bs', len( b1_k1),len( b2_k1))


           # B_A1 = B_A1 + + self.σ ** (-2) * self.k_t1 @ np.transpose(self.k_t1)


            #B_A1 = np.vstack([self.B_λi, np.transpose(b1_k1)+self.squared_exp_kernel(self.h, self.l, u_k1, u_k1)] )
            #self.k_t1 = np.vstack([self.k_t1,self.squared_exp_kernel(self.h, self.l,u_k1, u_k1)] )
           # B_A1 = B_A1 + self.σ ** (-2) * self.k_t1 @ np.transpose(self.k_t1)


            #self.B_λi = np.hstack([B_A1, B_A2])
            #             self.mu_λs = self.σ ** (-2) * np.transpose(self.K_us) @ (self.B_λ) @ (self.C)

            #self.mu_λs = self.σ ** (-2) * np.transpose(self.K_us) @ (self.B_λ) @ (self.C)
           # var_λs = self.K_ss + self.K_su @ ((self.B_λ) - np.linalg.pinv(self.K_uu)) @ self.K_us
            #self.B_λ =np.linalg.inv( np.hstack([B_A1, B_A2]))

            # TODO: calculate more effeciently using properties of block inversion

            au1 = self.squared_exp_kernel(self.h, self.l, u_k1, self.U)
            au3 = self.squared_exp_kernel(self.h, self.l, u_k1, u_k1)
            Au = np.vstack([self.K_uu, au1])
            Bu = np.vstack([np.transpose(au1), au3])
            self.K_uu = np.hstack([Au, Bu])

            self.K_ux = np.vstack([self.K_ux, self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data)])

            #self.k_t1 = np.vstack([self.k_t1, self.squared_exp_kernel(self.h, self.l, u_k1, x_t)])




            self.K_us = np.vstack([self.K_us, self.squared_exp_kernel(self.h, self.l, u_k1, X_test)])
            self.K_su = np.transpose(self.K_us)



            a1= self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data) @ \
                self.delta @ self.squared_exp_kernel(self.h, self.l, self.X_data, self.U)

            a3 = self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data) @ \
                 self.delta @ self.squared_exp_kernel(self.h, self.l, self.X_data, u_k1)

            A = np.vstack([self.KK,a1])
            B = np.vstack([np.transpose(a1),a3])

            self.KK = np.hstack([A,B])




            self.C = np.vstack([self.C, self.squared_exp_kernel(self.h, self.l, u_k1, self.X_data)
                                @ self.delta @ self.Y_data])

            self.mu_λs = self.σ ** (-2) * np.transpose(self.K_us) @ (self.B_λ) @ (self.C)



            #print(mu_λs)
            #var_λs = self.K_ss + self.K_su @ ((self.B_λ) - np.linalg.pinv(self.K_uu)) @ self.K_us

            #Remove points
            if self.remove_flag==1:
                print('Removed Point')
                self.U = np.delete(self.U, 0)
                self.U = self.U.reshape(len(self.U), 1)
                self.C = np.delete(self.C , 0)
                self.C =self.C.reshape(len(self.C),1)
                self.KK = np.delete(self.KK, 0,axis=1)
                self.KK = np.delete(self.KK, 0, axis=0)
                self.K_uu = np.delete(self.K_uu, 0,axis=1)
                self.K_uu = np.delete(self.K_uu, 0, axis=0)
                self.K_ux = np.delete(self.K_ux, 0, axis=0)
                self.K_us = np.delete(self.K_us, 0, axis=0)
                self.K_su = np.delete(self.K_su, 0, axis=0)
                # #
                self.B_λi = np.delete(self.B_λi, 0, axis=1)
                self.B_λi = np.delete(self.B_λi, 0, axis=0)
                self.B_λ = np.delete(self.B_λ, 0, axis=1)
                self.B_λ = np.delete(self.B_λ, 0, axis=0)
            # update U vector
            self.U = np.vstack([self.U, u_k1])
            # self.U = np.delete(self.U, 0, axis=0)
            self.U = self.U.reshape(len(self.U), 1)

            #TODO:
            # 1. Variance calculation (DONE)
            # 2. Removal of inducing points (problem with equation B_lambda)
            # 3. Windowing of datapoints
            # 4. Comparision with Dense GP and Sparse GP
        return self.mu_λs, self.var_λs,self.U