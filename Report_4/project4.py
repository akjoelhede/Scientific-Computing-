#Scientific Computing, project 4.1, Ordinary Differential Equations
#written by Linea S Hedemark, qrg977

#import libraries
import numpy as np
import matplotlib.pyplot as plt

#Define starting parameters
#homosexual men
P1 = 5
X1 = 0.01

#bisexual men
P2 = 5
X2 = 0

#heterosexual women
Q = 100
Y = 0

#heterosexual men
R = 100
Z = 0

#step size
dt = 0.001

#make a function that computes the differentials, to make Euler and Runge-Kutta functions cleaner
def Diff(X, params):
    #function takes:
    # X: vector of variables, np.array([x1, x2, y, z])
    # params: vector of all parameters, np.array([p1, p2, q, r, a1, a2, b1, b2, b3, c1, c2, d1, e, r1, r2, r3, r4])
    #purpose: return all four differentials for given input
    x1, x2, y, z = X
    p1, p2, q, r, a1, a2, b1, b2, b3, c1, c2, d1, e, r1, r2, r3, r4 = params

    nextX1 = x1 + ( a1*x1*(p1-x1) + a2*x2*(p1-x1) + e*(p1-x1) - r1*x1)
    nextX2 = x2 + ( b1*x1*(p2-x2) + b2*x2*(p2-x2) + b3*y*(p2-x2) + e*(p2-x2)- r2*x2)
    nextY = y + ( c1*x2*(q-y) + c2*z*(q-y) + e*(q-y) - r3*y )
    nextZ = z + ( d1*y*(r-z) + e*(r-z) - r4*z)

    #save 
    nextX = np.array([nextX1, nextX2, nextY, nextZ])
    
    return nextX


### Euler Method ###
def Euler_step(X, params, step_size = dt):
    #function takes:
    # X: vector of variables, np.array([x1, x2, y, z])
    # params: vector of all parameters, np.array([p1, p2, q, r, a1, a2, b1, b2, b3, c1, c2, d1, e, r1, r2, r3, r4])

    #calculate differentials
    diffs = Diff(X, params)

    #calculate next step using forward Euler method
    nextX = X + diffs*step_size

    return nextX

### Runge-Kutta Method (4th order) ###
def Runge_Kutta_step(X, params, step_size = dt):
    #calculate k coefficients
    k1 = Diff(X, params)
    k2 = Diff(X + (step_size/2)*k1, params)
    k3 = Diff(X + (step_size/2)*k2, params)
    k4 = Diff(X + step_size*k3, params)

    nextX = X + (step_size / 6 ) * (k1 + 2*k2 + 2*k3  +k4)
    return nextX
