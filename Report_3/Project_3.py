import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from LJhelperfunctions import *
import timeit
import time

"Q1_____________________________________________________________"

Epsilon = 0.997
Sigma = 3.401
A = 4*Epsilon*Sigma**12
B = 4*Epsilon*Sigma**6

def potential1D(r, r0=0):
	V = A/(r-r0)**12 - B/(r-r0)**6
	return V

def potential_total(r, A=A, B=B):
	if len(r.shape) == 1:
		r = r.reshape((-1, 3))
	R2 = pdist(r, metric='sqeuclidean')
	V = np.sum(A/R2**6 - B/R2**3)
	return V


r1 = np.linspace(3, 11.0, num=500)
r0 = 0
p1 = potential1D(r1, r0)

fig, (ax, ax2) = plt.subplots(nrows=2)
ax.plot(r1, p1, label='V > 0')
#ax.plot(r1[p1 < 0], p1[p1 < 0], label='V < 0')
r2 = np.linspace(3, 5, 100)
p2 = potential1D(r2, r0)
ax2.plot(r2, p2, label='V > 0')
ax2.plot(r2[p2 < 0], p2[p2 < 0], label='V < 0')
ax.set_xlabel('Distance $r$')
ax.set_ylabel('Potential $V$')
ax2.set_xlabel('Distance $r$')
ax2.set_ylabel('Potential $V$')
ax.legend()
ax2.legend()
ax.set_title('Lennard-Jones potential between two atoms')
fig.tight_layout()
plt.show()

"Q2_____________________________________________________________"

def f(x):
	y = x**2-4*np.sin(x)
	return y

def Newton_solver1(f, x0, h= 5e-2, max_iter=50, tol=1e-3):
	x = [x0]

	for i in range(max_iter):
		fx= f(x[-1])
		fminus = f(x[-1]-h)
		fplus  = f(x[-1]+h)
		diff = (fplus-fx)/h
		x_new = x[-1]-fx/(diff)
		x.append(x_new)
		res = abs((x[-1]-[-2])/[-2])
		if res <= tol:
			break

	return np.array(x)

def Bisection(f, a, b, tol = 1e-6):
	a, b = min(a,b), max(a,b)
	fa = f(a)
	fb = f(b)
	if fa/abs(fa) == fb/abs(fb):
		return None
	while b-a > tol:
		m = a + (b-a)/ 2
		fa = f(a)
		fm = f(m)
		sfa = fa/abs(fa)
		fsm = fm/abs(fm)
		if sfa == fsm:
			a = m
		else:
			b = m
	return m
