import numpy as np
import matplotlib.pyplot as plt


def plot_samples(mu,cov,num_samples,x_s):
    # Draw samples from the Gaussian Process
    samples = np.random.multivariate_normal(mu, cov, size=num_samples)


    for i in range(num_samples):
        plt.plot(x_s, samples[i, :], alpha=0.7, label='Sample {}'.format(i + 1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    return

def plot_gp(y,x,x_s,mu,cov, mu_s, cov_s):


    # Plotting the first figure with data, mean, and uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Data')
    plt.scatter(x[len(x)-1]+1, mu, color='blue', label='predicted')

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
    plt.figure(figsize=(10, 6))

    # Plotting the second figure with individual samples
    num_samples = 6
    plot_samples(mu,cov,num_samples,x_s)


    # Display both figures
    plt.show()
