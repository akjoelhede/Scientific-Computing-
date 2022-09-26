import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')

f = x**2-4*sp.sin(x)

fderivative = f.diff(x)

print(fderivative)

def Newton_solver(f, x_0, max_iter):

	x_store = []

	for i in range(max_iter):
		x_0 = x_0 - float(f.evalf(subs={x:x_0}))  /float(fderivative.evalf(subs={x:x_0}))
		x_store.append(x_0)

	return x_store

print(Newton_solver(f, 3, 10))