

import numpy as np
import matplotlib.pyplot as plt

def plot_gp_dynamic(y, t, x_s, mu, cov, mu_s, cov_s):
    # Select the last 100 elements



    plt.rcParams["figure.figsize"] = (10, 4)

    # Clear the current figure

    plt.clf()
    # Plotting the figure with data, mean, and uncertainty

    plt.scatter(t, y , color='red', label='Data')
    plt.scatter(t[len(t)-1]+1, mu , color='blue', label='prediction')


    plt.xlabel('x')
    plt.ylabel('y')

    # Set the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Update the x-axis limits to slide the view range
    # Set the desired step size for sliding the x-axis range
    step_size = 0.1



    # Set the x-axis limits
    #plt.xlim(x_min, x_max)

    # Update the plot
    plt.pause(0.01)  # Pause to allow the plot to update
