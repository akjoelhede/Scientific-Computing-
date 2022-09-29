import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from LJhelperfunctions import *
import timeit
import time

"Q1_____________________________________________________________"

V_function = LJ(sigma=SIGMA, epsilon=EPSILON)

def Pot_two_particles(x):

	#Now we define the coordinates for each particle
	x0 = np.array([x,0,0])
	x1 = np.array([0,0,0])

	#Calculate the potential 
	points = np.stack((x0,x1))
	pot = V_function(points)

	return pot

def Pot_four_particles(x):

	#Now we define the coordinates for each particle
	x0 = np.array([x,0,0])
	x1 = np.array([0,0,0])
	x2 = np.array([14,0,0])
	x3 = np.array([7,3.2,0])

	#Calculate the potential 
	points = np.stack((x0,x1,x2,x3))
	pot = V_function(points)

	return pot

# Let x range from 3 to 11, and make empty array to store strenght of potential in
x_arr = np.linspace(3,11,100)
V2_arr = np.zeros_like(x_arr) 
V4_arr = np.zeros_like(x_arr) 

# Calculate potential for each x
for x, i in zip(x_arr, np.arange(len(x_arr))):

	# Two particles
	potential2 = Pot_two_particles(x) #Calculates pot for each x
	V2_arr[i] = potential2 # Inserts value into the ith place in the empty pot arrays 

	# Four particles
	potential4 = Pot_four_particles(x)
	V4_arr[i] = potential4

# Plot it
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(x_arr, V2_arr, 'C0o--', label='Two particles')
ax.plot(x_arr, V4_arr, 'C1o--', label='Four particles')
ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$V_{LJ}$', fontsize=15)
ax.set_title('Lennard-Jones Potential', fontsize=18)
ax.legend(fontsize=15)
plt.show()

"Q2_____________________________________________________________"

def Bisection(f, a, b, tol = 1e-13):
	a, b = min(a,b), max(a,b)
	fa = f(a)
	fb = f(b)
	if fa/abs(fa) == fb/abs(fb):
		return None

	n_calls = 2 

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

		n_calls += 1

	return m, n_calls

x, n_calls = Bisection(Pot_two_particles, 2, 6)

print(f'The root was found to be {x}, with {n_calls} calls. Sigma is {SIGMA}')

"Q3_____________________________________________________________"

def dPot_two_particles(r):
	y = 4*EPSILON*((6*SIGMA**6)/r**7 - (12*SIGMA**12)/r**13)
	return y

def Newton_solver1(f, df, x0, tol=1e-12, max_iter=100):
	
	x, n_calls = x0, 0 

	for i in range(max_iter):

		fx = f(x)
		
		if abs(fx) < tol:

			n_calls += 1
			return x, n_calls
		
		diff = df(x)
		x = x - fx/diff

		n_calls += 2


	return x, n_calls

x,n_calls = Newton_solver1(Pot_two_particles, dPot_two_particles, 2, tol = 1e-12)

print(f'The root was found to be {x}, with {n_calls} calls. Sigma is {SIGMA}')

"Q4_____________________________________________________________"

def Bisection_step(f, a, b, n_calls):

	m = a + (b-a)/ 2
	fa = f(a)
	fm = f(m)
	sfa = fa/abs(fa)
	fsm = fm/abs(fm)
	if sfa == fsm:
		a = m
	else:
		b = m

	n_calls += 2

	return a, b, n_calls

def Newton_Raphson_step(fx, dfx, x0):

	x = x0 - fx/dfx

	return x

def Frankenstein_root_finder(f, df , x0, a, b, tol):

	m = x0
	fm = f(m)
	n_calls = 1

	while abs(fm) > tol:

		dfm = df(m)
		n_calls += 100

		if dfm == 0:
			a, b, n_calls = Bisection_step(f, a, b, n_calls)

			m = a + (b-a)/2
		
		else:

			m = Newton_Raphson_step(fm, dfm, m)

			if m > a and m < b:

				fa = f(a)
				n_calls += 1 

				sfa = fa/abs(fa)
				fsm = fm/abs(fm)
				if sfa == fsm:

					a = m

				else:
					b = m

			else:
				a, b, n_calls = Bisection_step(f, a, b, n_calls)

				m = a + (b-a)/2

		fm = f(m)
		n_calls += 1

	return m, n_calls