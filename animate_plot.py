import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from PIL import ImageSequence
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
    plt.savefig('animation_frame.png')

    # Open the saved image
    image = Image.open('animation_frame.png')

    #image.show()
    frames.append(image)


    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration*10, loop=0)

