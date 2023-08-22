import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')


"""
Python script for visualisation of GPR

Author: Hakim Amer
Date: May 15, 2023
"""

def gen_sine(x,f,mean,A,noise=0):

    y = A*np.sin(f*x)+noise
   # y = A * np.sin(f * x) + mean + 0.5 * np.cos(2 * f * x) + 0.1 * np.sin(10 * f * x)
    #y = np.sin(f*x) + x**2 + np.cos(10*f*x)
    return y


#def gen_sine(x,f,mean,A):

#    y = A*np.sin(f*x) + mean + 0.5*np.cos(2*f*x) + 0.1*np.sin(10*f*x)
    #y = np.sin(f*x) + x**2 + np.cos(10*f*x)
#    return y

def gen_wave_paper(x,f,A,noise):

    y = np.sin(2*np.pi*f*x)
    #y = np.sin(f*x) + x**2 + np.cos(10*f*x)
    return y


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
iteration= -1
#ax1.legend()
def plot_static(X_data,Y_data,X_test,mu,var,R_tot ,R_m, U_n, Y_U,x_t,y_t,color_p,g_t, name,clear_plt,moving,Y_data_syn,X_data_syn):

    #plt.title(name)

    #ax1.plot(X_data, Y_data, color='cyan', label='Data',alpha =0.5, linewidth=3)
    ax1.scatter(U_n, U_n*0.0, color='pink', label='inducing-points', marker ='x',alpha =0.5, linewidth=3)
    ax1.scatter(x_t, y_t,[150], color='purple', label='new data', marker ='*',alpha =0.5, linewidth=3)

    ax1.scatter(X_data_syn, Y_data_syn, [1], color='cyan', label='ground truth')




    ax1.plot(X_test, mu[:,0], color=color_p, label=name,alpha =0.5, linewidth=3)
    # ax1.fill_between(
    #      X_test[:,0],
    #      mu[:,0] - np.sqrt(np.diag(var)),
    #      mu[:,0] + np.sqrt(np.diag(var)),
    #      color=color_p,
    #      alpha=0.13,
    #      label='Uncertainty'
    #  )

    ax1.set_xlabel('t')
    ax1.set_ylabel('f(t)')
    ax1.set_xlim([ X_test[len(X_test)-1]-2*np.pi,X_test[len(X_test)-1]+1,])
    ax1.set_xlim([0.0, 2 * np.pi])

    if moving==1:
       #ax1.set_xlim([ x_t-1*np.pi,X_test[len(X_test)-1]+0.5])
       ax1.set_xlim([x_t-0.5, x_t+1])
       ax1.set_xlim([0.0, x_t + 1])
       ax2.set_xlim([ x_t-1*np.pi,X_test[len(X_test)-1]+0.5])
       ax3.set_xlim([x_t - 1 , X_test[len(X_test) - 1] + 0.5])
       ax1.set_ylim([-2,2])
       ax2.set_ylim([0, 0.5])
    #ax1.set_xlim([0.0, 2*np.pi ])
    #ax2.set_ylim([-0.3,0.3 ])


    #Errors Plot

    errors = abs(Y_data_syn[(int(x_t*100) -10):(int(x_t*100)  +10)] - mu)
    ax2.plot(X_test, errors, color_p, label='Error',alpha =0.5, linewidth=3)
    ax2.set_xlabel('t')
    ax2.set_ylabel('Prediction Error')
    ax2.legend()


    #ax2.text(x_t+5-1.18,4, str( 'error ='),bbox=dict(facecolor='purple', alpha=0.1))
    #ax2.text(x_t+5,4, str(errors[1]),bbox=dict(facecolor='purple', alpha=0.1))
    #ax3.legend()

    ax2.vlines(x=x_t, ymin=-1, ymax=1,
               colors='purple',
               label='current time', linestyles='dashed')





    #Relevance of inducing points plot
    #ax3.plot(U_n[0:len(U_n)-1], R_m,'blue', label='R_m')
    #ax3.bar(U_n[0:len(U_n)-1],R_m, color = 'blue',linewidth=0.1, sethatch=0)
    #ax3.bar(U_n[0:len(U_n)-1],R_m, width=0.8, bottom=None, align='center', data=None)

    #ax3.bar(U_n[0:len(U_n)-1], R_m,width = 0.001)
   # ax3.set_xlabel('t')
   # ax3.set_ylabel('R_m')
   # ax3.legend()

   # ax3.text(x_t+1,2, str( 'R_tot ='),bbox=dict(facecolor='red', alpha=0.1))
   # ax3.text(x_t+2.2,2, str(R_tot[0]),bbox=dict(facecolor='red', alpha=0.1))
    #plt.tight_layout()

    #ax3.clear()
    # Display the plot
    #plt.show(block=False)
    plt.show(block=False)

    plt.pause(0.001)
    #ax1.clear()
    if clear_plt==1:
       ax1.clear()
       ax2.clear()
       ax3.clear()

    return

