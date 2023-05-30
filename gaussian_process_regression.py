import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

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





    def squared_exp_kernel(self,h, delta, x1, x2):
        """
        Isotropic squared exponential kernel.

        Args:+
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).

        Returns:
            (m x n) matrix.
        """
        #sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)

       # K = h ** 2 * np.exp(-0.5 / l ** 2 * sqdist)

        #multivariate
        K1 = np.ones((len(x1),len(x2)))
        K = np.ones((len(x1),len(x2)))
        #l1=60
        #delta = np.diag([l,l1])
        n_inputs = np.shape(x1)[1]
        '''
        for kk in range(n_inputs):
            for i in range(len(x1)):
                for j in range(len(x2)):
                    #K1[i,j] = (h ** 2)  * np.exp((-(x1[i,kk] - x2[j,kk]) ** 2 )/ (2 * delta[kk,kk] ** 2))
                    K1[i, j] = (h ** 2) * np.exp((-(x1[i, kk] - x2[j, kk]) ) @ delta[kk, kk]
            K = K * K1
        '''
        #diffx=np.zeros((len(x1),len(x2)))
        #for i in range(len(x1)):
        #    for j in range(len(x2)):
        #        diffx[i,j] = x1[i,:]-x2[j,:]
       # d = np.sqrt()
        #diffx= sum(diffx)


        delta = np.diag(delta)
        #delta[1,1]=0.0
        diffx =x1[:, None]-x2
        a = diffx@np.linalg.pinv(delta)

        #dist = a[:] @ diffx[:]
        #dist = np.dot(a[:],  diffx[:])
        dist=np.zeros((len(x1),len(x2)))
        for i in range(len(x1)):
            for j in range(len(x2)):
                dist[i,j]= np.dot(diffx[i,j,:].T,  np.linalg.pinv(delta)@diffx[i,j,:])



        K = h* np.exp(-dist)
        a=1
        #delta[0, 0]=0.3
        #delta[1,1]=10000
       # sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
       # K = h ** 2 * np.exp(-0.5 / delta[0,0] ** 2 * sqdist)
        sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1)
        delta_inv_sq = np.linalg.pinv(delta)**2
        delta_inv_sq = np.linalg.pinv(delta) ** 2
        sqdist = sqdist - 2 * np.dot(x1, delta_inv_sq @ x2.T)
        #malh= - 0.5 * delta_inv_sq @sqdist

        #K = h ** 2 * np.exp(-0.5 * sqdist)


       # K = h ** 2 * np.exp(-0.5 / l ** 2 * sqdist)

        #a=1
        KK = (h ** 2) * np.exp((-(x1[:, None] - x2) ** 2) / (2 * delta[0] ** 2))
        return K


    def generate_sine(self,period,amplitude,x,n_data, n_ind,n_test, noise):
       # if x == []:
       #     x = np.linspace(0, 2 * np.pi, n_data)

        #x_s = np.linspace(0, x[len(x)-1], n_test)
        u = np.linspace(0, 2 * np.pi, n_ind)
        #xn = np.array([x[len(x)-1]+0.001])
        xn=0
        x_s=0
        yn=0
        y = amplitude * np.sin(period * x) + noise * np.random.randn(*x.shape)
        #yn = np.array([y[len(y) - 1] + 0.001])


        return [y,u,xn,yn]

    def example_system(x0, n):

        x = np.array(np.zeros(n))
        x[0] = x0
        for i  in range(len(x)-1):
            x[i+1] = x[i]/2 + (25 * x[i] / (1 + (x[i]) ** 2)) * np.cos(x[i]) + np.random.normal(0,1)


        return x


    def basic_gp(self,h,l,mu_0, y, x, x_s):
        K_xx =  self.squared_exp_kernel(h, l, x, x)
        K_x_x =  self.squared_exp_kernel(h, l, x_s, x)
        K_xx_ =  self.squared_exp_kernel(h, l, x, x_s)
        K_x_x_ =  self.squared_exp_kernel(h, l, x_s, x_s)


        K_inv = np.linalg.pinv(K_xx)
        mu = K_x_x @ K_inv @ y
        cov = K_x_x_ - K_x_x @ K_inv @ K_xx_


        return mu, cov

    def negative_log_likelyhood(self,X_train, Y_train, noise, naive=True):
        """
        Returns a function that computes the negative log marginal
        likelihood for training data X_train and Y_train and given
        noise level.

        Args:
            X_train: training locations (m x d).
            Y_train: training targets (m x 1).
            noise: known noise level of Y_train.
            naive: if True use a naive implementation of Eq. (11), if
                   False use a numerically more stable implementation.

        Returns:
            Minimization objective.
        """
        #Y_train = Y_train.ravel()
        #X_train = X_train.ravel()

        #Y_train = np.sum(Y_train, 1)

        def nll_naive(theta):
            # Naive implementation of Eq. (11). Works well for the examples
            # in this article but is numerically less stable compared to
            # the implementation in nll_stable below.
            #h_th = theta[1]
            h_th =theta[0]
            l_th = theta[1:]
           # l_th  =np.diag( theta[1:])
            K = self.squared_exp_kernel( h_th,l_th, X_train, X_train)  + \
            noise**2 * np.eye(len(X_train))
            #K = np.squeeze(K)

            #b =0.5 * np.log(np.linalg.det(K)) + \
            #    0.5 * Y_train.dot(np.linalg.inv(K).dot(Y_train)) + \
            #    0.5 * len(X_train) * np.log(2 * np.pi)
            #a = theta
            #0.5 * Y_train.T.dot(np.linalg.inv(K).dot(Y_train))  # Transpose Y_train using .T
            a=1
            #K= np.squeeze(K)
            a = 0.5 * Y_train.T@(np.linalg.pinv(K)@Y_train)
            b=1
           # print(K)
           # print(theta[0])
           # print(theta[1])
            #0.5 * Y_train.dot(np.linalg.inv(K.T).dot(Y_train))  # Transpose np.linalg.inv(K) using .T
            return 0.5 * np.log(np.linalg.det(K)) + \
                (0.5 * Y_train.T@(np.linalg.inv(K)@Y_train)).ravel() + \
                0.5 * len(X_train) * np.log(2 * np.pi)
            # 0.5 * Y_train.dot(np.linalg.inv(K).dot(Y_train)) + \

        def nll_stable(theta):
            # Numerically more stable implementation of Eq. (11) as described
            # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
            # 2.2, Algorithm 2.1.
            #h_th = theta[0]
            #l_th = theta[1]
            #l_th = np.diag(theta[1:])
            h_th =theta[0]
            l_th = theta[1:]
            K = self.squared_exp_kernel( h_th,l_th, X_train, X_train)  + \
                noise  * np.eye(len(X_train))

           # K= np.round(K, 1)
           # eps = 1e-02
           # K[K <= 0] = eps

            L = np.linalg.cholesky(K)

            S1 = solve_triangular(L, Y_train, lower=True)
            S2 = solve_triangular(L.T, S1, lower=False)

            return np.sum(np.log(np.diagonal(L))) + \
                0.5 * Y_train.ravel().dot(S2) + \
                0.5 * len(X_train) * np.log(2 * np.pi)

        if naive:
            return nll_naive
        else:
            return nll_stable

    # Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
    # We should actually run the minimization several times with different
    # initializations to avoid local minima but this is skipped here for
    # simplicity.
    def hyper_param_opt(self,X_train, Y_train,h0,l0, noise):
        d= X_train.shape[1]
       # d =1
        theta_0 = 0.1 * np.ones(d+1)
        #theta_0 = [l0,h0]

        bounds = [(0.01, 10)] * (d+1)
        res = minimize(self.negative_log_likelyhood(X_train, Y_train, 0.0), theta_0,
                       bounds=bounds,
                       method='L-BFGS-B',   options={'gtol': 1e-18})

        # Store the optimization results in global variables so that we can
        # compare it later with the results from other implementations.
        #sigma_f_opt = res.x[1]
        l_opt = res.x[1:]
        sigma_f_opt = res.x[0]

        return [sigma_f_opt,l_opt]




    def log_max_likelihood(self,h,l,X_data,Y_data,noise):
        K = self.squared_exp_kernel( h,l, X_data, X_data)+ \
            noise ** 2 * np.eye(len(X_data))
        #J = -0.5 *Y_data.T @np.linalg.inv(K)@ Y_data + \
            #-0.5 * np.log (np.linalg.det(K)) + \
            #- 0.5 * len(X_data) * np.log(2*np.pi)

        L = np.linalg.cholesky(K)

        S1 = solve_triangular(L, Y_data, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)

        J = np.sum(np.log(np.diagonal(L))) + \
            0.5 * Y_data.T.dot(S2) + \
            0.5 * len(X_data) * np.log(2 * np.pi)

        return J

    def grad_K(self, h,l,x1,noise):
        grad_K_sigma =2 * h * (-(x1[:, None] - x1) ** 2 )/ (2 * l ** 2)
        grad_K_l =  h**2 * np.exp((-(x1[:, None] - x1) ** 2 )/ (2 * l ** 2)) *(-(x1[:, None] - x1) ** 2 )/ ( l ** 3)

        return [np.squeeze(grad_K_sigma),np.squeeze(grad_K_l)]
    def grad_J (self,h,l,X_data,Y_data,dK_dsig,dK_dth,noise):
        K = self.squared_exp_kernel(h, l, X_data, X_data) +\
              noise ** 2 * np.eye(len(X_data))
        alpha = np.linalg.inv(K) @ Y_data
        dJ_dsig = 0.5 * np.trace(((alpha @ alpha.T - np.linalg.inv(K)))@dK_dsig)
        dJ_dl = 0.5 * np.trace(((alpha @alpha.T - np.linalg.inv(K))) @dK_dth)
        return [dJ_dsig,dJ_dl]
    def update_params_g_descent(self,h_i,l_i,X_data,Y_data,opt_iter,learn_rate,eps,noise):
        for i in range(opt_iter):
            [dK_dsig,dK_dth] = self.grad_K(h_i, l_i, X_data,noise)
            [dJ_dsig, dJ_dl]=self.grad_J(h_i, l_i, X_data, Y_data, dK_dsig, dK_dth,noise)
            J1 = self.log_max_likelihood(h_i,l_i,X_data,Y_data,noise)
           # h_i = h_i + learn_rate * dJ_dsig
            h_i = h_i - 0.0001* learn_rate * dJ_dsig
            l_i = l_i - learn_rate * dJ_dl
            J2 = self.log_max_likelihood(h_i,l_i,X_data,Y_data,noise)
            res=abs(J1-J2)
            if res<eps:
                break
        h_opt= h_i
        l_opt = l_i
        return[ h_opt,l_opt,res]










    def log_max_likelyhood_simple(self,x, y, noise):
        def step(theta):
            delta = theta[1:]
            #delta = np.diag(delta)

            a=1
            K =  self.squared_exp_kernel(theta[0], delta, x, x)+\
              noise ** 2 * np.eye(len(x))

            a1=  0.5 * np.log(np.linalg.det(K))
            a2=  0.5 * y.T @ np.linalg.pinv(K) @ y
           # a3 = np.sum(np.log(np.diagonal(np.linalg.cholesky(K))))
            a4 = 0.5 * y.T @ np.linalg.pinv(K) @ y


            a=  0.5 * np.log(np.linalg.det(K)) + \
                   0.5 * y.T @ np.linalg.pinv(K) @ y + \
                   0.5 * len(x) * np.log(2*np.pi)
            #a= 0.5 * np.log(np.linalg.det(K)) + \
            #        0.5 * y.T @ np.linalg.pinv(K) @ y + \
            #        0.5 * len(x) * np.log(2*np.pi)
            a= a.ravel()
            b=1
            return a


        return step

    
    def hyper_param_optimize_simple(self,h0,l0,x, y,noise):
        d= x.shape[1]
        theta_0 = 1* np.ones(d+1)
        res = minimize( self.log_max_likelyhood_simple(x, y, noise), theta_0,
                   method='L-BFGS-B',bounds=((1e-5, 10), (1e-5, 10)))


        sigma_f_opt = res.x[0]
        l_opt = res.x[1:]
        return l_opt, sigma_f_opt
