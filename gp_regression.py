import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plot_utils import plot_gp
from plot_utils_dynamic import plot_gp_dynamic

from animate_plot import plot_gp_animation
from get_drone_data import calculate_force_estimates_and_obtain_data
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

    cov_u = Kuu @ np.linalg.inv(Kuu + Kuf @ np.linalg.pinv(delta_ff)@ Kfu) @ Kuu
    mu_u = m_u + cov_u @ np.linalg.pinv(Kuu) @ Kuf @ np.linalg.pinv(delta_ff) @ y

    mu_s = m_s + Ksu @  np.linalg.pinv(Kuu) @ mu_u
    cov_s = Kss - Ksu @ np.linalg.pinv(Kuu) @ (Kuu - cov_u) @ np.linalg.pinv(Kuu) @ Kus
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
        a =1
        #return np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
        #       0.5 * y.T @ np.linalg.pinv(K) @ y + \
        #       0.5 * len(x) * np.log(2*np.pi)
        return -0.5 * y.T @K @y - 0.5 * np.log(np.linalg.det(K)) - n/2 * np.log (2* np.pi)
    return step

# minimize -log liklihood

def hyper_param_optimize(x, y):
    res = minimize(log_max_likelyhood(x, y, 0.0), [1, 1],
               bounds=((1e-20, None), (1e-20, None)),
               method='L-BFGS-B')

    l_opt, sigma_f_opt = res.x
    return l_opt, sigma_f_opt



def squared_exp_fun(h, l, x1, x2):
    #K = (h ** 2)  * np.exp((-(x1[:, None] - x2) ** 2 )/ (2 * l ** 2))
    K = np.zeros([np.shape(x1)[1],np.shape(x1)[1]])
    if (x1.ndim > 1) or (x2.ndim > 1):
        for i in range(np.shape(x1)[0]):
            print(i)
            K = K + (h ** 2) * (np.exp((-(x2[i, :] - np.tile(x1[i, :], (x2.shape[1], 1)).T) ** 2) / (2 * l ** 2)))

        #K = np.sum((h ** 2) * np.exp((-(x2[:, :] - np.tile(x1[:, :], (x2.shape[1], 1)).T) ** 2) / (2 * l ** 2)), axis=1)
    else:
        K = (h ** 2)  * np.exp((-(x1[:, None] - x2) ** 2 )/ (2 * l ** 2))


    return K

def multi_input_squared_exp_fun(h, delta, x1, x2):
    #K = (h ** 2)  * np.exp((-0.5 *  (x1 - x2).T @ delta @ (x1 - x2) ))
    squared_dist = np.sum((x1.T[:, None] - x2) ** 2, axis=-1)
    K = (h ** 2) * np.exp(-squared_dist / (2 * delta ** 2))

    return K

def generate_sine(period,amplitude,x):
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


def basic_gp_2(h,l,mu_0, y, x, x_s):



    K_xx = multi_input_squared_exp_fun(h, l, x, x)
    K_x_x = multi_input_squared_exp_fun(h, l, x_s, x)
    K_xx_ = multi_input_squared_exp_fun(h, l, x, x_s)
    K_x_x_ = multi_input_squared_exp_fun(h, l, x_s, x_s)

    K_inv = np.linalg.inv(K_xx)
    mu = mu_0 + K_x_x @ K_inv @ y
    cov = K_x_x_ - K_x_x @ K_inv @ K_xx_


    return mu, cov
def basic_gp(h,l,mu_0, y, x, x_s):
    K_xx = squared_exp_fun(h, l, x, x)
    K_x_x = squared_exp_fun(h, l, x_s, x)
    K_xx_ = squared_exp_fun(h, l, x, x_s)
    K_x_x_ = squared_exp_fun(h, l, x_s, x_s)



    K_inv = np.linalg.pinv(K_xx)
    mu = K_x_x @ K_inv @ y
    cov = K_x_x_ - K_x_x @ K_inv @ K_xx_


    return mu, cov

if __name__ == '__main__':
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

    [y,x, x_s,u,xn,yn]= generate_sine(period, amplitude,[])
    step_size =  (x[2] - x[1])

    #Optimize hyperparameters of the GP
    #[l, h] = hyper_param_optimize(x, y)

    #Compute and Vizualize
    sim_time =300
    run_anim= 0
    frames = []
    if run_anim ==1:
        for i in range (sim_time):

            x_s = np.linspace(0, x[len(x)-1]+pred_ahead, n_test )

            #[mu, cov] = basic_gp(h, l, mu_0, y, x, x_s)
            [mu, cov] = offline_sparse_gp_FITC(h, l, mu_0, y, x, x_s, u)
            #plot_gp(y,x,x_s,mu,cov, mu, cov)

            it=i
            plot_gp_animation(y, x, x_s, mu, cov, mu, cov, 'animation.gif', sim_time,it,frames)
            #x_end = x[len(x)-1] + step_size

            x = np.append(x, x[-1] + step_size)
            off_u = 0.01 #offset to prevent x=u
            u = np.append(u, u[-1] + step_size - off_u )

            u = u[1:]
            x = x[1:]
            x_s = x_s[1:]
            y = generate_sine(period, amplitude,x)[0]


    #Plotting
    h = 1
    #l = np.diag([1,1])









# drone data

[gt_x, gt_y, gp_pr_x, gp_pr_y, pos_x, vel_x] = calculate_force_estimates_and_obtain_data()
sim_time =300
n_i=200
n=10

skip = 1
dist_a = gt_x[n_i::skip]
x2 = [0]
x2 = np.append(x2, np.array(np.diff(gt_x[n_i::skip])))



x1_i = gt_x[1::skip]
x1_i = np.array(dist_a[(1):(n_i)])

x2_i = [0]
x2_i = np.append(x2, np.array(np.diff(gt_x[1::skip])))
x2_i = x2_i[0:(len(x1_i))]
x_i = np.array([x1_i,x2_i])
x_i = x2_i

y_i = dist_a[(2):(n_i+1)]
#x_i = np.linspace(0, len(y_i), len(y_i))
h=10

#l =np.array([1,1])
l=1

#[l, h] = hyper_param_optimize(x_i, y_i)
print(l)
print(h)
for i in range (sim_time):
    x1 = np.array(dist_a[(1+i):(n+i)])
    x2 = x2[0:(len(x1))]
    x = np.array([x1,x2])


    y = dist_a[(2+i):(n+1+i)]
    x_s1= np.array([y[len(y)-1]])
    x_s2 = np.array([y[len(y) - 1]- y[len(y) - 2]])
    x_s = np.array([x_s1,x_s2])
    [mu, cov] = basic_gp(h, l, mu_0, y, x, x_s)

    t= np.linspace(i+n_i, len(y)+i, len(y))


    t_s = np.linspace(i, len(mu)+i, len(mu)+i)



    plot_gp_dynamic(y, t,  t, mu, cov, mu, cov)
