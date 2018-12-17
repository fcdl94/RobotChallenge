from numpy import exp, arange, maximum
import numpy as np
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

P_d = 1
A_d = 50
A_b = 60
A_f = 90

def score_fun_1(a,p):
    return 1000* 2 * ((1 / (1 + p)) - 1/2) * (maximum(0, a - A_b) / (A_f - A_b))


def score_func_2d(a, p, pow):
    beta = maximum(0, a - A_d)
    return [(1/pow) * pow**(P_d/p) * alpha(A_d) * b**2 for b in beta]

def score_decathlon(e):
    return beta(A_d) * maximum(0, A_d - e)**2

def score_func_3d(a, p, pow):
    beta = maximum(0, a - A_d)
    return pow**(P_d/p-1) * alpha(A_d) * beta**2

def beta(e_max):
    return 1000 * e_max**(-2)

def alpha(a_min):
    return 1000 * ((100 - a_min) ** (-2))

_3D = False

if _3D:
    pm = arange(2, 2, 0.02)
    acc = arange(50, 100, 1)
    X, Y = meshgrid(acc, pm)  # grid of point
    Z = score_fun_1(X, Y)  # evaluation of the function on the grid
    
    fig = plt.figure(figsize=(4,4), dpi=200)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                          cmap=cm.RdBu,linewidth=0, antialiased=False)
    
    ax.set_zlim(0, 1200)
    ax.set_xlim(50, 100)
    ax.set_ylim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.xaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.ylabel('parameters, $p_i^m$')
    plt.xlabel('accuracy')
    #plt.title('$ 5 \\cdot {2}^{\\frac{P_d}{p_m}} \\cdot \\alpha_d  \\cdot max (0, A_d - A_{min})^2, A_{min}=' + str(A_d) + '$')
    #plt.title(' $ 2 \\cdot ( \\frac{P_d}{p} - \\frac{1}{2}) * ( \\frac{a - A_b}{A_f - A_b}) $ ')
    plt.show()

else:
    fig, ax = plt.subplots(figsize=(4,4), dpi=200)
    pm = arange(1, 2.01, 0.02)
    acc = arange(50, 110, 10)
    #err = np.arange(0, A_d, 1)
    Z = score_func_2d(acc, pm, 10)
    #Z = score_decathlon(err)
    plt.ylabel('Score')
    #plt.xlim(50,100)
    plt.xlabel('Parameters')
    #plt.title('$ 10 \\cdot {1}^{\\frac{P_d}{p_m}} \\cdot \\alpha_d  \\cdot max (0, A_d - A_{min})^2, A_{min}=' + str(A_d) + '$')
    #plt.title('$ alpha_d  \\cdot max (0, E_{max} - E_d)^2, E_{max}=' + str(A_d) + '$')
    
    plt.plot(pm, Z[0], label='Acc=50')
    plt.plot(pm, Z[1], label='Acc=60')
    plt.plot(pm, Z[2], label='Acc=70')
    plt.plot(pm, Z[3], label='Acc=80')
    plt.plot(pm, Z[4], label='Acc=90')
    plt.plot(pm, Z[5], label='Acc=100')
   
    #for a in pm:
    #    plt.plot(acc, score_fun_1(acc, a),  label='Par='+"{:.2f}".format(a))
   
    #plt.plot(err, Z, [10], score_decathlon(np.array([10])), "rd")
    plt.legend()
    plt.show()