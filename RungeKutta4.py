import numpy as np

#v = [t,x0,v0]
#f = lambda v: v[2]
#g = lambda v: -(k/m) * v[1]
#F = [f, g]
# f ~ v - velocity , g ~ acceleration
#K = [t+h, x0 + f*h, v0 + g*h]

def RK4(v, F, h):
    n = len(F)
    K = [[0]*(len(F)+1)]
    for i in range(3):
        L = [1]
        for ii in range(n):
            l = np.array(K[-1], dtype=object)
            L += [F[ii](v+l*(h/2))]
        K += [L]

    L = [1]
    for ii in range(n):
        l = np.array(K[-1], dtype=object)
        L += [F[ii](v + l * h)]
    K += [L]

    for i in range(4):
        K[i] = np.array(K[i], dtype=object)
    Kvec = np.array(K[1] + 2*K[2] + 2*K[3] + K[4], dtype=object)

    return v + (1/6)*Kvec*h

