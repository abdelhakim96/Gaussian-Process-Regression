import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate data
x = np.linspace(0, 2 * np.pi, 10)
y = np.sin(x)
x_s = np.linspace(0, 2 * np.pi, 100)

# Step 2: Define GP parameters
h = 1
l = 1

# Calculation of covariance function K(xi, xj)
K_xx = h * np.exp(-(x[:, None] - x) ** 2 / l ** 2)
K_x_x = h * np.exp(-(x_s[:, None] - x) ** 2 / l ** 2)
K_xx_ = h * np.exp(-(x[:, None] - x_s) ** 2 / l ** 2)
K_x_x_ = h * np.exp(-(x_s[:, None] - x_s) ** 2 / l ** 2)

# Compute the mean and covariance of the posterior
K_inv = np.linalg.inv(K_xx)
mean_posterior = K_x_x @ K_inv @ y
cov_posterior = K_x_x_ - K_x_x @ K_inv @ K_xx_

# Draw 6 samples from the Gaussian Process
num_samples = 6
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
