
#I learned from my friend Nikolai Nielsen how to import from other variables and functions from other files 
# and i figured that i would use that to since the code will be much nicer to look at
from watermatrices import Amat, Bmat, yvec, Imat
from subroutines import *
from HHexamples import *
import numpy as np
import matplotlib.pyplot as plt

E = np.vstack((np.hstack((Amat, Bmat)), np.hstack((Bmat, Amat))))

S = np.vstack((np.hstack((Imat, np.zeros_like(Imat))), np.hstack((np.zeros_like(Imat), -Imat))))

z = np.array([*yvec, *-yvec])

diag = np.array((*[1]*7, *[-1]*7))
S = np.diag(diag)



omega = np.array([0.800,1.146,1.400])
domega = 0.5*10**(-3)


print("\nAnswer a2:")
for i in omega:
	M = E-i*S
	Cond_num = cond(M)
	print(f"Condition number for each omega = {i}")
	print(Cond_num)

print("\nAnswer b1:")
for i in omega:
	e = error_bound(E, S, i)
	print(f"Error bound for omega={i}:")
	print(e)

M = np.array([[2, 1, 1], [4, 1, 4], [-6, -5., 3]])
b = np.array([4, 11, 4.])

print("\nAnser c1:")
if lu_solve(M,b).all() == np.linalg.solve(M, b).all():
	print("it worked !!! I knew you had it in you")
	print(lu_solve(M,b))
	print(np.linalg.solve(M, b))
else:
	print("I have a bad feeling about this, something is wrong")


print("\nAnswer d1:")

for i in omega:
	a = solve_alpha(i, E, S, z)
	print("\nAnswer to d1.1")
	print(f"alpha(omega) of omega {i}")
	print(a)

	up_bound = solve_alpha(i+domega, E, S, z)
	low_bound = solve_alpha(i-domega, E, S, z)

	print("\nAnswer to d1.2")
	print(f"alpha(omega+deltaomega) for omega{i}")
	print(up_bound)

	print("\nAnswer to d1.3")
	print(f"alpha(omega-deltaomega) for omega{i}")
	print(low_bound)

e1_interval = np.linspace(0.7,1.5,1000)
alpha_e1 = np.zeros(e1_interval.shape)

for n in range(e1_interval.size):
        alpha_e1[n] = solve_alpha(e1_interval[n], E, S, z)

plt.plot(e1_interval,alpha_e1)
plt.xlabel("Omega")
plt.ylabel("Alpha(omega)")
plt.savefig("e1.pdf")
print("\n")

#Week 2 

#F1
#Check that QR decomposition works

print("\n Answer to F")
Q, R = qr_factorize(A2)

identity_check = Q.T@Q
R_check = Q@R

print(identity_check)
print("\n")
print(R_check)

x = least_squares(A2,b2)

print("\n Slow Householder QR decomposition on matrix A2")
print(f'Resulting R matrix: R= \n {R}')
print("\n Least Squares method on matrix A2 and vector b2")
print(f'linear least square fit: x={x}')


#F2


#F3



#G

omega_p = 1.1 # select appropriate Omega_p < 1.5 
omega = np.linspace(0.7,1.5,1000) #Makes am array of omegas
omega = omega[omega < omega_p] # Shrink to the omegas thats gonna be in use
alpha = np.zeros(omega.shape) #constructs an array that has the same size as omega for when solving for alpha

for i in range(omega.size):
	alpha[i] = solve_alpha(omega[i],E,S,z) # solving alpha for each omega in line 115

#Solves the least squares problem for the given polynomial for both n = 4 & 6 
x_1, p_1 = least_squares_P(omega, alpha, 4) 
x_2, p_2 = least_squares_P(omega, alpha, 6)

#print(x_1,x_2)

#Calculates the relative error 
rel1 = np.abs((p_1-alpha)/alpha)
rel2 = np.abs((p_2-alpha)/alpha)

#Plot relative wrt. omega in a logarithmic scale
fig2, (ax2, ax3) = plt.subplots(ncols=2, figsize=(8, 4))
ax2.plot(omega, rel1, label='n=4')
ax2.plot(omega, rel2, label='n=6')
ax2.set_yscale('log')
ax2.set_xlabel(r'$\omega$')
ax2.set_ylabel(r'$\log_{10} |(P(\omega) - \alpha(\omega))/\alpha(\omega)|$')
ax2.legend()

#Utilizes that -log10(relative_error) = sig. digits and then uses np.floor to get an integer
ax3.plot(omega, np.floor(-np.log10(rel1)), label='n=4')
ax3.plot(omega, np.floor(-np.log10(rel2)), label='n=6')
ax3.legend()
ax3.set_xlabel(r'$\omega$')
ax3.set_ylabel(r'Number of significant digits of $P(\omega)$')
fig2.tight_layout()
fig2.savefig('g.pdf')

"____________________________________________________________________________"

#Solution to h 
 
omega = np.linspace(0.7,1.5,1000) #Makes am array of omegas
alpha = np.zeros(omega.shape) #constructs an array that has the same size as omega for when solving for alpha

for i in range(omega.size):
	alpha[i] = solve_alpha(omega[i],E,S,z) # solving alpha for each omega in line 115

#Solves the least squares problem for the given polynomial for both n = 2 & 4 
params_1, Q1 = least_squares_Q(omega, alpha, 2) 
params_2, Q2 = least_squares_Q(omega, alpha, 4)

#print(params_1,params_2)

#Calculates the relative error 
rel1 = np.abs((Q1-alpha)/alpha)
rel2 = np.abs((Q2-alpha)/alpha)

#Plot relative wrt. omega in a logarithmic scale
fig2, (ax2, ax3) = plt.subplots(ncols=2, figsize=(8, 4))
ax2.plot(omega, rel1, label='n=2')
ax2.plot(omega, rel2, label='n=4')
ax2.set_yscale('log')
ax2.set_xlabel(r'$\omega$')
ax2.set_ylabel(r'$\log_{10} |(P(\omega) - \alpha(\omega))/\alpha(\omega)|$')
ax2.legend()

#Utilizes that -log10(relative_error) = sig. digits and then uses np.floor to get an integer
ax3.plot(omega, np.floor(-np.log10(rel1)), label='n=2')
ax3.plot(omega, np.floor(-np.log10(rel2)), label='n=4')
ax3.legend()
ax3.set_xlabel(r'$\omega$')
ax3.set_ylabel(r'Number of significant digits of $Q(\omega)$')
fig2.tight_layout()
fig2.savefig('h.pdf')


