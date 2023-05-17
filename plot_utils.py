import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

"""
Python script for visualisation of GPR

Author: Hakim Amer
Date: May 15, 2023
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
figs_dir = os.path.join(script_dir, 'figs')
root_dir = os.path.dirname(figs_dir)

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


def plot_gp_dynamic(y_train, x_train, x_test, t_train,t_test, mu, cov, pred_ahead):
    # Select the last 100 elements



    plt.rcParams["figure.figsize"] = (18, 6)

    # Clear the current figure

    plt.clf()
    # Plotting the figure with data, mean, and uncertainty

    plt.scatter(t_train, y_train , color='red', label='Past Data',marker='o')

    plt.plot(t_test, mu , color='blue', label='prediction')

    '''
    plt.fill_between(
        t_test,
        mu - np.sqrt(np.diag(cov)),
        mu + np.sqrt(np.diag(cov)),
        color='gray',
        alpha=0.4,
        label='Uncertainty'
    )
    '''


    plt.xlabel('x')
    plt.ylabel('y')

    # Set the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Update the x-axis limits to slide the view range
    # Set the desired step size for sliding the x-axis range
    step_size = 0.1


    # Set the x-axis limits
    plt.xlim(t_train[0], t_train[len(t_train)-1] + pred_ahead)

    # Update the plot
    #plt.pause(1)  # Pause to allow the plot to update
    plt.show()

def plot_gp_animation(y, x, x_s, mu, cov, mu_s, cov_s, filename, duration,iter,frames):
    # Select the last 100 elements
    last_100_mu = mu[-100:]
    last_100_cov = cov[-100:]
    last_100_mu_s = mu_s[-100:]
    last_100_x_s = x_s[-100:]
    last_100_cov_s = cov_s[-100:]

    # Select the last 10 elements
    last_100_y = y
    last_100_x = x

    # Clear the current figure
    plt.clf()

    # Plotting the figure with data, mean, and uncertainty
    plt.scatter(last_100_x, last_100_y, color='red', label='Data')
    plt.plot(last_100_x_s, last_100_mu_s, color='blue', label='Predicted Mean')

    plt.fill_between(
        last_100_x_s,
        last_100_mu - np.sqrt(np.diag(last_100_cov)),
        last_100_mu + np.sqrt(np.diag(last_100_cov)),
        color='gray',
        alpha=0.4,
        label='Uncertainty'
    )
    plt.xlabel('x')
    plt.ylabel('y')

    # Set the legend outside the plot
    plt.legend(loc='upper left')

    # Update the x-axis limits to slide the view range
    # Set the desired step size for sliding the x-axis range
    step_size = 0.1

    last_100_x_s = last_100_x_s[-100:]  # Limit x_s to the last 100 points
    x_min = max(last_100_x_s[-1] - 2 * np.pi, -2 * np.pi)  # Set the minimum x-axis limit
    x_max = last_100_x_s[-1]  # Set the maximum x-axis limit

    # Set the x-axis limits
    plt.xlim(x_min, x_max + 0.02)
    plt.ylim(-1.1, 1.5)

    # Save the plot as an image
    plt.savefig(os.path.join(figs_dir, 'animation_frame.png'))
    # Open the saved image
    image_path = os.path.join(figs_dir, 'animation_frame.png')
    image = Image.open(image_path)
    #image.show()
    frames.append(image)
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration*10, loop=0)