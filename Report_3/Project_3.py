import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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

print(Newton_solver1(f, 3))

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

print(Bisection(f, 1, 4))