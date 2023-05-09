import numpy as np
import matplotlib.pyplot as plt




def plot_gp(y,x,x_s,mu,cov, mu_s, cov_s):
    # Draw 6 samples from the Gaussian Process
    num_samples = 10
    samples = np.random.multivariate_normal(mu, cov, size=num_samples)

    # Plotting the first figure with data, mean, and uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Data')


    #plt.plot(x_s, mu, color='blue', label='Predicted Mean')
    plt.plot(x_s, mu_s, color='blue', label='Predicted Mean')

    plt.fill_between(
        x_s,
        mu - np.sqrt(np.diag(cov)),
        mu + np.sqrt(np.diag(cov)),
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
