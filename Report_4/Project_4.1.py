#Anders Terp Kjoelhede
#Date: 21/10/2022
#Scientific Computing
#Project_4

from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

dt = 0.001

x1 = 0.01
x2 = 0.01
y = 1
z = 1

a1 = 10
a2 = 5
b1 = 5
b2 = 1
b3 = 1
c1 = 1
c2 = 1
d1 = 1
p1 = 5
p2 = 5
r = 100 
q = 100 
e = 0
r1 = 0
r2 = 0
r3 = 0
r4 = 0

var = np.array([x1, x2, y, z])

parameters = np.array([a1, a2, b1, b2, b3, c1, c2, d1, p1, p2, r, q, e, r1, r2, r3, r4])

def Diff(Var, parameters):

	x1, x2, y, z = Var
		
	a1, a2, b1, b2, b3, c1, c2, d1, p1, p2, r, q, e, r1, r2, r3, r4 = parameters

	diffx1 = a1*x1*(p1-x1)+ a2*x2*(p1-x1) + e*(p1-x1) - r1*x1

	diffx2 = b1*x1*(p2-x2)+b2*x2*(p2-x2) + b3*y*(p2-x2) + e*(p2-x2)- r2*x2

	diffy = c1*x2*(q-y) + c2*z*(q-y) + e*(q-y) - r3*y

	diffz = d1*y*(r-z) + e*(r-z) - r4*z

	next = np.array([diffx1, diffx2, diffy, diffz])

	return next

#CALL TO DIFF FUNCTION
def euler(Var, parameters, dt):

	diff = Diff(Var, parameters)

	euler_step = Var + diff * dt

	return euler_step

def Runge_Kutta_step(X, params, step_size = dt):
	#calculate k coefficients
	k1 = Diff(X, params)
	k2 = Diff(X + (step_size/2)*k1, params)
	k3 = Diff(X + (step_size/2)*k2, params)
	k4 = Diff(X + step_size*k3, params)

	nextX = X + (step_size / 6 ) * (k1 + 2*k2 + 2*k3  +k4)
	return nextX

euler_store = [var]
for i in range(299):
	new_step = euler(euler_store[-1], parameters, dt)
	euler_store.append(new_step)
euler_store = np.array(euler_store)


RK_store = [var]
for i in range(299):
	new_step = Runge_Kutta_step(RK_store[-1], parameters, dt)
	RK_store.append(new_step)
RK_store = np.array(RK_store)

t = np.arange(0,300*dt, dt)

plt.plot(t, euler_store[:,0], label = "Homosexual males")
plt.plot(t, euler_store[:,1], label = "Bisexual males ")
plt.plot(t, euler_store[:,2], label = "Heterosexual females")
plt.plot(t, euler_store[:,3], label = "Heterosexual males")
plt.hlines(y = 5, xmin = 0, xmax = 0.3, linestyles = "dashed", color = "k", label = "x1=x2=5")
plt.hlines(y = 100, xmin = 0, xmax = 0.3, linestyles = "dashed", color = "k", label = "p1=p2=100")

plt.title("HIV infection with forward euler")
plt.xlabel("Time")
plt.ylabel("Number of infected")

plt.legend()
plt.show()

plt.plot(t, RK_store[:,0], label = "Homosexual males")
plt.plot(t, RK_store[:,1], label = "Bisexual males ")
plt.plot(t, RK_store[:,2], label = "Heterosexual females")
plt.plot(t, RK_store[:,3], label = "Heterosexual males")
plt.hlines(y = 5, xmin = 0, xmax = 0.3, linestyles = "dashed", color = "k", label = "x1=x2=5")
plt.hlines(y = 100, xmin = 0, xmax = 0.3, linestyles = "dashed", color = "k", label = "p1=p2=100")

plt.title("HIV infection with 4th order Runge-Kutta")
plt.xlabel("Time")
plt.ylabel("Number of infected")

plt.legend()
plt.show()