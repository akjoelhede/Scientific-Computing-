# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from LJhelperfunctions import *
import timeit
import time

# %%
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
ax.plot(x_arr, V2_arr, '-o', label='Two particles')
ax.plot(x_arr, V4_arr, '-o', label='Four particles')
ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$V_{LJ}$', fontsize=15)
ax.set_title('Lennard-Jones Potential', fontsize=18)
ax.legend(fontsize=15)
plt.show()

# %%
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

# %%
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

# %%
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
		n_calls += 1

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

# %%
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

# %%
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

# %%
#*"QG_____________________________________________________________"

def golden_section(f, a, b, tol = 1e-3):

	#Golden ration ish 
	tau = (np.sqrt(5)-1)/2

	#We need to compute the initial points x1 and x2 within the interval of [a,b] from tau
	x1 = a + (1-tau)*(b-a)
	f1 = f(x1)

	x2 = a + tau*(b-a)
	f2 = f(x2)

	#Start of counter
	n_calls = 2

	#* MAIN FUNCTION

	while abs(b-a) > tol:

		#CHECK which subinterval contains the minima and discard the other

		#RIGHT interval
		if f1 > f2:

			#Shorten interval
			a = x1

			#Change midpoints
			x1 = x2
			f1 = f2

			#Then find x2 from tau
			x2 = a + tau*(b-a)
			f2 = f(x2)

		# else LEFT interval
		else:

			#Shorten interval
			b = x2

			#Change midpoints
			x2 = x1
			f2 = f1

			#Calculate x1 from tau
			x1 = a + (1-tau)*(b-a)
			f1 = f(x1)

		n_calls += 1

		x_optimized = a + (b-a) / 2

	return x_optimized, n_calls

#We can then test this on f) to see if we obtain the same alpha

f = Line_function(f = flat_V, x0 = x0.flatten(), d = -gradient_pot_func(x0).flatten())

alpha, n_calls = golden_section(f, 0, 1)

print(f'The minima found with the golden section function is {alpha} with {n_calls} calls')

# Next we use this to find the distance between two argon atoms

r, n_calls = golden_section(Pot_two_particles, 2, 6)

print(f'The distance between the two atoms is {r}, with {n_calls} calls')

# %%
#*QH_____________________________________________________________"

def BFGS(f, gradf, x0, tol = 1e-6, max_iter = 10000):

	#Initial Guess
	x = x0

	#Identity matrix
	I = np.eye(len(x0))

	#Initial hessian approximation
	B_inv = I

	#initial gradient
	y = gradf(x)

	n_calls = 1

	#This status is kept until max_iter is hit, then change to false
	converged = True

	#*MAIN function 

	while np.linalg.norm(y) > tol and n_calls < max_iter:

		#STEP 1: obtain a direction 
		p = - np.dot(B_inv,y)

		#STEP 2: Calculate the new position
		x_new = x + p

		#STEP 3: Calculate the new gradient
		y_new = gradf(x_new)

		dy = y_new - y 

		n_calls += 1

		#STEP 4: Get new inverse Hession matrix

		#Make calculation more manageable
		bfgs_1 = I - (p[:, np.newaxis] * dy[np.newaxis, :])/(np.dot(dy, p))
		bfgs_2 = I - (dy[:, np.newaxis] * p[np.newaxis, :])/(np.dot(dy, p))

		#Calculate the new Inverse hessian matrix
		B_inv_new = np.dot(bfgs_1, np.dot(B_inv, bfgs_2)) + (p[:, np.newaxis] * p[np.newaxis, :])/(np.dot(dy, p))

		#FINAL STEP: Update B_inv, x, y
		B_inv, x, y = B_inv_new, x_new, y_new

	#Update convergence 
	if n_calls >= max_iter:
		converged = False

	#Optimized x value
	x_opt = x 

	return x_opt, n_calls, converged

# load data
data = np.load("ArStart.npz")
X_start2 = data["Xstart2"]

#find positions
x, n_calls, conv = BFGS(flat_V,flat_gradV, X_start2, tol = 1e-6, max_iter=100)
positions = x.reshape(2, -3)
print(f'Minimum found to be at x0 = {positions[0]} and x1 = {positions[1]} with {n_calls} calls to the function, convergence = {conv}')

#Find distance
distances = distance(positions)
print(f'The distance is found to be {distances[0,1]}')

#%%
#*"QI_____________________________________________________________"

names =['Xstart2', 'Xstart3', 'Xstart4', 'Xstart5', 'Xstart6', 'Xstart7', 'Xstart8', 'Xstart9']

r = 3.817

fig, ax = plt.subplots(ncols = 4, nrows = 2, figsize = (15,8), subplot_kw=dict(projection='3d'))
ax = ax.flatten()

for i in range(len(names)):

	X_start = data[names[i]]

	N = i + 2

	x, n_calls, conv = BFGS(flat_V, flat_gradV, X_start, tol = 1e-10)
	positions = x.reshape(N, -3)

	print(f'For {i+1} particles the function was called {n_calls} times. Convergence = {conv}')


	if conv == True:
		distances = distance(positions)

		one_percent = sum(abs(distances-r)/r <= 0.01)

		print(f'For {N} particles, {one_percent} were within 1% of the two particle optimum {r}')

		ax[i].scatter(positions[:,0], positions[:,1], positions[:,2], color = 'b')
		ax[i].set_title(f'{i+2} Particles')
# %%
#*"QJ_____________________________________________________________"

def BFGS_line_seach(f, gradf, x0, tol = 1e-6, max_iter = 10000):

	#Initial Guess
	x = x0

	#Identity matrix
	I = np.eye(len(x0))

	#Initial hessian approximation
	B_inv = I

	#initial gradient
	y = gradf(x)

	n_calls = 1

	#This status is kept until max_iter is hit, then change to false
	converged = True

	#*MAIN function 

	while np.linalg.norm(y) > tol and n_calls < max_iter:

		#STEP 1: obtain a direction 
		p = - np.dot(B_inv,y)


		#STEP 1.5: Implement line search
		f_1D = Line_function(f, x, p) #Get the function of a line segment
		alpha, n_calls_extra = golden_section(f_1D, -1, 1, tol=1e-6) #Put that funtion into the golden section search to estimate alpha
		n_calls += n_calls_extra # Add the extra calls to our calls


		#STEP 2: Calculate the new position
		x_new = x + alpha * p # Introduce alpha into the calculation of the new position

		#STEP 2.5: Get the displacement
		s = x_new - x 

		#STEP 3: Calculate the new gradient
		y_new = gradf(x_new)

		dy = y_new - y 

		n_calls += 1

		#STEP 4: Get new inverse Hession matrix

		#Make calculation more manageable
		bfgs_1 = I - (s[:, np.newaxis] * dy[np.newaxis, :])/(np.dot(dy, s))
		bfgs_2 = I - (dy[:, np.newaxis] * s[np.newaxis, :])/(np.dot(dy, s))

		#Calculate the new Inverse hessian matrix
		B_inv_new = np.dot(bfgs_1, np.dot(B_inv, bfgs_2)) + (s[:, np.newaxis] * s[np.newaxis, :])/(np.dot(dy, s))

		#FINAL STEP: Update B_inv, x, y
		B_inv, x, y = B_inv_new, x_new, y_new

	#Update convergence 
	if n_calls >= max_iter:
		converged = False

	#Optimized x value
	x_opt = x 

	return x_opt, n_calls, converged

# load data
data = np.load("ArStart.npz")
X_start2 = data["Xstart2"]

#find positions
x, n_calls, conv = BFGS_line_seach(flat_V, flat_gradV, X_start2, tol = 1e-6, max_iter=100)
positions = x.reshape(2, -3)
print(f'Minimum found to be at x0 = {positions[0]} and x1 = {positions[1]} with {n_calls} calls to the function, convergence = {conv}')

#Find distance
distances = distance(positions)
print(f'The distance is found to be {distances[0,1]}')

#* CALCULATE AND PLOT

names =['Xstart2', 'Xstart3', 'Xstart4', 'Xstart5', 'Xstart6', 'Xstart7', 'Xstart8', 'Xstart9']

r = 3.817

fig, ax = plt.subplots(ncols = 4, nrows = 2, figsize = (15,8), subplot_kw=dict(projection='3d'))
ax = ax.flatten()

for i in range(len(names)):

	X_start = data[names[i]]

	N = i + 2

	x, n_calls, conv = BFGS_line_seach(flat_V, flat_gradV, X_start, tol = 1e-5, max_iter = 100000)
	positions = x.reshape(N, -3)

	print(f'For {i+1} particles the function was called {n_calls} times. Convergence = {conv}')

	if conv == True:
		distances = distance(positions)

		one_percent = sum(abs(distances-r)/r <= 0.01)

		print(f'For {N} particles, {one_percent} were within 1% of the two particle optimum {r}')

		ax[i].scatter(positions[:,0], positions[:,1], positions[:,2], color = 'b')
		ax[i].set_title(f'{i+2} Particles')

#todo"QK_____________________________________________________________"



# %%
