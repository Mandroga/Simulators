import numpy as np
from numpy import random

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

from math import pi
from math import cos
from math import sin
from math import atan
from math import sqrt

from RungeKutta4 import RK4

#Constants ---------


#GConstant
G = 2*6.67184*(10**(-20))
#AstronomicalUnit
Au = 149597871
#MassConstant
MC = 10**24
#dayseconds
ds = 24*3600

time_multiplier = 1
h = ds/4 * time_multiplier
tf = 365 * ds * time_multiplier
t = 0

Sun = np.array([0, np.array([0, 0]), np.array([0, 0])], dtype=object)
Mercury = np.array([0, np.array([0, 0.39*Au]), np.array([-45.9, 0])], dtype=object)
Venus = np.array([0, np.array([0.72*Au, 0]), np.array([0, 32])], dtype=object)
Earth = np.array([0, np.array([-1*Au, 0]), np.array([0, -25.8])], dtype=object)
Meteor = np.array([0, np.array([-3*Au, -3*Au]), np.array([10, 15])], dtype=object)

M=[1989000*MC, 0.33*MC, 4.87*MC, 5.97*MC, 0.1*MC]
R = [Sun, Mercury, Venus, Earth, Meteor]

P = [ [[], []] for i in range(len(M))]

#i = 0
#k = 0

fr = lambda v: v[2]
fvi = lambda v: (G*M[i]) * (R[i][1]-v[1]) / (np.linalg.norm(R[i][1]-v[1])**3)

F = [fr, fvi]

while t <= tf:

    for k in [*range(len(M))]:
        otherplanets = [*range(len(M))]
        otherplanets.pop(k)
        for i in otherplanets:
            R[k] = RK4(R[k], F, h)

    for k in [*range(len(M))]:
        P[k][0] += [R[k][1][0]]
        P[k][1] += [R[k][1][1]]

    t += h

fig = plt.figure()
l = 6
fig.set_figwidth(l)
fig.set_figheight(l)

#abs(max(r1, key=lambda t: abs(t)))+50
xl = 4*Au
yl = xl
plt.xlim(-xl, xl)
plt.ylim(-yl, yl)


masses = [ plt.plot([], [], 'o') for u in range(len(M))]
trajectories = [ plt.plot([], [], 'red') for u in range(len(M))]

def animate (i):

    for k in range(len(M)):
        masses[k][0].set_data(P[k][0][i], P[k][1][i])
        #trajectories[k][0].set_data(P[k][0][:i], P[k][1][:i])

    return masses

ani = FuncAnimation(fig, animate, frames=int(tf/h), interval=1)
plt.grid()
plt.show()
