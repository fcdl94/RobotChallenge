from numpy import exp, arange, maximum
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

P_d = 1
A_d = 50


def score_func(a, p):
    beta = maximum(0, a - A_d)
    return (10/2.71) * 2.71**(P_d/p) * alpha(A_d) * beta**2


def alpha(a_min):
    return 1000 * ((100 - a_min) ** (-2))


pm = arange(1, 3, 0.05)
acc = arange(0, 100, 0.5)
X, Y = meshgrid(acc, pm)  # grid of point
Z = score_func(X, Y)  # evaluation of the function on the grid


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.set_zlim(0, 10000)
ax.zaxis.set_major_locator(LinearLocator(11))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.ylabel('parameters scaled by $P_d$')
plt.xlabel('accuracy')
plt.title('$ \\frac{10}{2.71} \\cdot {2.71}^{\\frac{P_d}{p_m}} \\cdot \\alpha_d  \\cdot max (0, A_d - A_{min})^2, A_{min}=' + str(A_d) + '$')
plt.show()