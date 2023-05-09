import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

'''

# gradient descent implementation attempt
def hyperparam_opt( h, l, iter, y, x, learning_r):
    C_h = 2 * np.exp((-(x - x) ** 2) / (2 * l ** 2))
    for i in range(iter):

        K = (h ** 2) * np.exp((-(x[:, None] - x) ** 2) / (2 * l ** 2))
        alpha = np.linalg.inv(K) * y
        dKdh =  h * C_h
        dKdl = 2 * h ** 2 * (x[:, None] - x) ** 2 * np.exp(-(x[:, None] - x) ** 2 / l ** 2) / l ** 3

        dlogdl = 0.5 * np.trace((alpha @ alpha.T - np.linalg.inv(K)) * dKdl)

        #K = squared_exp_fun(h, l, x, x)
        #[dKdh, dKdl] = derivative_kernel(x, x, h, l)
        #[dlogdh, dlogdl] = dlog_likelihood(y, K, dKdh, dKdl)
        #obj = log_likelihood(y, K, dKdh, dKdl)


        #print(dKdh)
        #print(dKdl)
        #e_h = learning_r * dlogdh
        #h = h + learning_r * dlogdh
        h= h
        l = l + learning_r * dlogdl
    return h, l


def dlog_likelihood(y, K, dKdh, dKdl):
    alpha = np.linalg.inv(K) @ y
    dlogdh = 0.5 * np.trace((alpha @ alpha.T - np.linalg.inv(K)) @ dKdh)
    dlogdl  = 0.5 * np.trace((alpha @ alpha.T - np.linalg.inv(K)) * dKdl)

    return dlogdh, dlogdl


def log_likelihood(y, K, dKdh, dKdl):
    n=10
    log_p = -0.5 * y.T @K @y - 0.5 * np.log(np.linalg.det(K)) - n/2 * np.log (2* np.pi)


    return -log_p


def derivative_kernel(x1, x2, h, l):
    dKdh = 2 * h * np.exp((-(x1[:, None] - x2) ** 2) / (2 * l ** 2))
    dKdl = 2 * h ** 2 * (x1[:, None] - x2) ** 2 * np.exp(-(x1[:, None] - x2) ** 2 / l ** 2) / l ** 3
   # dKdl = (h ** 2) * np.exp((-(x1[:, None] - x2) ** 2) * (1 / (2 * l ** 2))) * (-((x1[:, None] - x2) ** 2) * (1 / (l ** 3)))
    return dKdh, dKdl
'''



def log_max_likelyhood(x, y, noise):
    def step(theta):
        K = squared_exp_fun(theta[1], theta[0], x, x)
        return np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
               0.5 * y.T @ np.linalg.inv(K) @ y + \
               0.5 * len(x) * np.log(2*np.pi)
    return step

# minimize -log liklihood

def hyper_param_optimize(x, y):
    res = minimize(log_max_likelyhood(x, y, 0.0), [1, 1],
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')

    l_opt, sigma_f_opt = res.x
    return l_opt, sigma_f_opt



def squared_exp_fun(h, l, x1, x2):
    K = (h ** 2)  * np.exp((-(x1[:, None] - x2) ** 2 )/ (2 * l ** 2))

    return K

# Step 1: Define GP parameters
mu = 0.0
iter = 2000
n_data = 10
period = 2
# Step 2: Generate data
x = np.linspace(0, 2 * np.pi, n_data)
y = np.sin(period  * x)
x_s = np.linspace(0, 2 * np.pi, 100)



# Calculation of covariance function K(xi, xj)
h = 10 #optimal around 1
l = 1000
learning_r = 0.0001



print(h)
print(l)

[l, h] = hyper_param_optimize(x, y)

# optimize
#[h, l] = hyperparam_opt(h, l, iter, y,x, learning_r)
print(h)
print(l)

K_xx = squared_exp_fun(h, l, x, x)
K_x_x = squared_exp_fun(h, l, x_s, x)
K_xx_ = squared_exp_fun(h, l, x, x_s)
K_x_x_ = squared_exp_fun(h, l, x_s, x_s)

#[dKdh, dKdl]= derivative_kernel(x, x, h, l)
#cost = log_likelihood(y, K_xx, dKdh, dKdl)

#print(cost)
# Compute the mean and covariance of the posterior
K_inv = np.linalg.inv(K_xx)
mean_posterior = mu + K_x_x @ K_inv @ y
cov_posterior = K_x_x_ - K_x_x @ K_inv @ K_xx_


# Draw 6 samples from the Gaussian Process
num_samples = 10
samples = np.random.multivariate_normal(mean_posterior, cov_posterior, size=num_samples)







# Plotting the first figure with data, mean, and uncertainty
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data')
plt.plot(x_s, mean_posterior, color='blue', label='Predicted Mean')
plt.fill_between(
    x_s,
    mean_posterior - np.sqrt(np.diag(cov_posterior)),
    mean_posterior + np.sqrt(np.diag(cov_posterior)),
    color='gray',
    alpha=0.4,
    label='Uncertainty'
)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Plotting the second figure with individual samples
plt.figure(figsize=(10, 6))
for i in range(num_samples):
    plt.plot(x_s, samples[i, :], alpha=0.7, label='Sample {}'.format(i + 1))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Display both figures
plt.show()

