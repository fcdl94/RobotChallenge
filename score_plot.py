from numpy import exp, arange, maximum
import numpy as np
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

P_d = 1
A_d = 50


def score_func_2d(a, p, pow):
    beta = maximum(0, a - A_d)
    return [(10/pow) * pow**(P_d/p) * alpha(A_d) * b**2 for b in beta]


def score_func_3d(a, p, pow):
    beta = maximum(0, a - A_d)
    return (10/pow) * pow**(P_d/p) * alpha(A_d) * beta**2


def alpha(a_min):
    return 1000 * ((100 - a_min) ** (-2))

_3D = True

if _3D:
    pm = arange(1, 4, 0.04)
    acc = arange(0, 100, 0.5)
    X, Y = meshgrid(acc, pm)  # grid of point
    Z = score_func_3d(X, Y, 2)  # evaluation of the function on the grid
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                          cmap=cm.RdBu,linewidth=0, antialiased=False)
    
    ax.set_zlim(0, 10000)
    ax.zaxis.set_major_locator(LinearLocator(11))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    
    plt.ylabel('parameters scaled by $P_d$')
    plt.xlabel('accuracy')
    plt.title('$ 5 \\cdot {2}^{\\frac{P_d}{p_m}} \\cdot \\alpha_d  \\cdot max (0, A_d - A_{min})^2, A_{min}=' + str(A_d) + '$')
    plt.show()

else:
    fig, ax = plt.subplots()
    pm = arange(1, 4, 0.04)
    acc = np.array([50, 60, 70, 80, 90, 100])
    Z = score_func_2d(acc, pm, 1)
    plt.ylabel('Score')
    plt.xlabel('Parameters')
    plt.title('$ 10 \\cdot {1}^{\\frac{P_d}{p_m}} \\cdot \\alpha_d  \\cdot max (0, A_d - A_{min})^2, A_{min}=' + str(A_d) + '$')
    
    plt.plot(pm, Z[0], label='Acc=50')
    plt.plot(pm, Z[1], label='Acc=60')
    plt.plot(pm, Z[2], label='Acc=70')
    plt.plot(pm, Z[3], label='Acc=80')
    plt.plot(pm, Z[4], label='Acc=90')
    plt.plot(pm, Z[5], label='Acc=100')
    
    plt.legend()
    plt.show()