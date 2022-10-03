import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from LJhelperfunctions import *
import timeit
import time

#*"QA_____________________________________________________________"

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

#*"QB_____________________________________________________________"

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

#*"QC_____________________________________________________________"

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

#*"QD_____________________________________________________________"

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

# Test it
x, n_calls = Frankenstein_root_finder(Pot_two_particles, dPot_two_particles, 2, 2, 6, tol=1e-13)

# Print results
print(f'The root was found to be {x} with {n_calls} calls. Sigma is {SIGMA}')


#*"QE_____________________________________________________________"

#Get the gradient potential function with the experimental values of argon
gradient_pot_func = LJgradient(sigma =SIGMA, epsilon = EPSILON)


#As before, define a function that calculates the gradient potential for two particles
def Gradient_pot_twoparticles(x):

	x0 = np.array([x,0,0])
	x1 = np.array([0,0,0])

	points = np.stack((x0,x1))
	gradient_pot = gradient_pot_func(points)

	return gradient_pot


x_arr = np.linspace(3,10,1000) #x is in the interval [3:10]

#Create empty arrays for the gradient in all directions
gradx_arr = np.zeros_like(x_arr)
grady_arr = np.zeros_like(x_arr)
gradz_arr = np.zeros_like(x_arr)

for x, i in zip(x_arr, np.arange(len(x_arr))): 

	grad = Gradient_pot_twoparticles(x)

	if i == 0:
		print(grad)
	
	gradx_arr[i], grady_arr[i], gradz_arr[i] = grad[0]


# Potenial needs to be in the same interval
pot_arr = np.zeros_like(x_arr) 

# Calculate potential for each x
for x, i in zip(x_arr, np.arange(len(x_arr))):
	potential = Pot_two_particles(x)
	pot_arr[i] = potential

# Plot the nonzero component for x0
fig, ax = plt.subplots(figsize=(8,6))

# Plot the gradient
ax.plot(x_arr, gradx_arr, '-', label='Gradient')

# Plot the potential 
ax.plot(x_arr,pot_arr,'-',label='LJ Potential')
ax.hlines(0,3,10,color='k', linestyle='dashed')
ax.vlines(3.81,-5,5,color='k', linestyle='dashed')

ax.set_title('Two Particle System', fontsize=14)
ax.set_xlabel(r'x-coordinate of particle $\mathbf{x_0}$', fontsize=14)
ax.legend(fontsize=14)

ax.set_ylim(-5,5)

plt.show()


#Do the same thing for four particles

#Four particle potential
def Gradient_pot_fourparticle(x):

	#coordinates for the particles
	x0 = np.array([x,0,0])
	x1 = np.array([0,0,0])
	x2 = np.array([14,0,0])
	x3 = np.array([7,3.2,0])

	# Calculate potential
	points = np.stack((x0,x1,x2,x3))
	gradV = gradient_pot_func( points )

	return gradV

x_arr = np.linspace(3,10,1000) #x is in the interval [3:10]

#Create empty arrays for the gradient in all directions
gradx_arr = np.zeros_like(x_arr)
grady_arr = np.zeros_like(x_arr)
gradz_arr = np.zeros_like(x_arr)


for x, i in zip(x_arr, np.arange(len(x_arr))): 

	grad = Gradient_pot_twoparticles(x)

	if i == 0:
		print(grad)

	gradx_arr[i], grady_arr[i], gradz_arr[i] = grad[0]

# Potenial needs to be in the same interval
pot_arr = np.zeros_like(x_arr) 

# Calculate potential for each x
for x, i in zip(x_arr, np.arange(len(x_arr))):
	potential = Pot_four_particles(x)
	pot_arr[i] = potential

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x_arr, gradx_arr, '-', label='gradient')
ax.plot(x_arr,pot_arr,'-',label='potential')
ax.hlines(0,3,10,color='k', linestyle='dashed')
ax.set_ylim(-5,5)
ax.legend(fontsize=14)
ax.set_title('Four Particle System', fontsize=14)
ax.set_xlabel(r'x-coordinate of particle $\mathbf{x_0}$', fontsize=14)

plt.show()

#*"QF_____________________________________________________________"

#The same old bisection solver for finding roots

def Flat_Bisection(f, a, b, tol = 1e-13):
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

#Defines a line segment and finds the function for a given line section
def Line_function(f, x0, d):

	def f_restriction(alpha):

		Line_segment = x0 + alpha * d

		return f(Line_segment)
		
	return f_restriction

#Calculates the directional derivative of f(x0 + alpha * d) by doing matrix multiplication
#With d and F  
def Directional_derivative_line(f, x0, d):
	f_gradient = Line_function(f, x0, d)

	def Directional_derivative(alpha):
		return d @ f_gradient(alpha)

	return Directional_derivative

#Finds the root of a given line segment using the above functions
def Line_search(F, x0, d, a, b, tol):

	#We need to flatten because we are working in 1D and because both
	#d and x0 are represented in 2D
	x0_flat = x0.flatten()
	d_flat = d.flatten()

	directional_derive_func = Directional_derivative_line(F, x0_flat, d_flat)


	alpha, n_calls = Flat_Bisection(directional_derive_func, a, b, tol = tol)

	return alpha, n_calls

#*## Test ###

#Initial guess
x0 = np.array([[4,0,0], [0,0,0], [14,0,0], [7,3.2,0]])

#Direction
d = -gradient_pot_func(x0)

#The interval of alpha
a, b = 0,1

#flat_gradV is given in the LJhelperfunctions.py
alp, calls = Line_search(flat_gradV, x0, d, a, b, tol = 1e-13)
print(f'Found alpha value to be {alp} with {calls} calls to the function')




#todo"QG_____________________________________________________________"





#todo"QH_____________________________________________________________"





#todo"QI_____________________________________________________________"




#todo"QJ_____________________________________________________________"




#todo"QF_____________________________________________________________"
