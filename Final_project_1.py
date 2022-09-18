from watermatrices import Amat, Bmat, yvec, Imat
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
z_sig_digits = 0.5*10e-8

"____________________________________________________________________________"


def max_norm(M):
	norm = np.max(np.sum(np.abs(M), axis=1))
	return norm


"____________________________________________________________________________"


def cond(M):
	M_inv = np.linalg.inv(M)
	M_norm = max_norm(M)
	M_inv_norm = max_norm(M_inv)
	condition_num = M_norm * M_inv_norm
	return condition_num


"____________________________________________________________________________"

def error_bound(E, S, omega):
	M = E-omega*S
	Cond_num = cond(M)
	Max_norm = max_norm(M)

	return Cond_num * max_norm(S*1/2*10**(-3))/ Max_norm



"____________________________________________________________________________"

def lu_factorize(M):
    
    #Get the number of rows 
	n = M.shape[0]
    
	U = M.copy()
	L = np.eye(n)
    
    #Loop over the rows 
	for i in range(n):
            
        #Eliminate entries below i with row operations 
        #on U and reverse the row operations to 
        #manipulate L
		factor = U[i+1:, i] / U[i, i]
		L[i+1:, i] = factor
		U[i+1:] -= factor[:, np.newaxis] * U[i]
	return L, U

"____________________________________________________________________________"


def forward_substitution(L, b):
    
	y = np.zeros(b.shape)
	for i in range(b.size):
		y[i] = b[i] - L[i, :i].dot(y[:i])
	return y


"____________________________________________________________________________"


def back_substitution(U, y):
    
	x = np.zeros(y.shape)
	for i in reversed(range(y.size)):
		x[i] = (y[i] - U[i, i:].dot(x[i:]))/U[i, i]
	return x


"____________________________________________________________________________"

def lu_solve(M, b):
    
	L, U = lu_factorize(M)
    
	y = forward_substitution(L, b)
    
	return back_substitution(U, y)

"____________________________________________________________________________"


def solve_alpha(omega, E, S, z):
	
	M = E-omega*S

	L,U = lu_factorize(M)
	y = forward_substitution(L,z)
	x = back_substitution(U,y)
	alpha = z.dot(x)

	return alpha 

"____________________________________________________________________________"


def make_householder(a):
    #find prependicular vector to mirror
    u = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    u[0] = 1
    H = np.eye(a.shape[0])
    #finding Householder projection
    H -= (2 / np.dot(u, u)) * np.outer(u,u)
    return H
"____________________________________________________________________________"

def householder_fast(M):
	R = M.copy()
	R = R.astype(float)
	m,n = R.shape
	V_store = []

	for i in range(n):
		a_vector = R[i:m,i]
		alpha = -np.copysign(a_vector[0],a_vector[0])/np.abs(a_vector[0])*(np.sqrt(np.sum(a_vector**2)))
		v_vector = a_vector.copy()
		v_vector[0] = v_vector[0]*alpha
		v_vector = v_vector/np.sqrt(np.sum(v_vector**2))
		V_store.append(v_vector)
		for j in range(i,n):
			R[i:m,j]=R[i:m,j]-2*np.dot(v_vector,R[i:m,j])*v_vector
		
	return R, V_store

"____________________________________________________________________________"


def qr_factorize(A):
    m, n = A.shape # Divide shape of M into m,n
    Q = np.eye(m) # make an identity matrix that the m big on each side
    for i in range(n - (m == n)):
        H = np.eye(m)
        #calculate Householder matrix i: rows and i: columns from A i: rows and ith column
        H[i:, i:] = make_householder(A[i:, i])
        Q = Q@H
        A = H@A
    return Q, A

"____________________________________________________________________________"

 
def least_squares(A, b):
    m, n = A.shape
    Q, R = qr_factorize(A)
    b2 = Q.T@b
    x = back_substitution(R[:n], b2[:n])

    return x

"____________________________________________________________________________"


def least_squares_P(x, y, n):
    m = x.size
    A = np.zeros((m, n+1))
    for j in range(n+1):
        A[:, j] = x**(2*j)
    res = least_squares(A, y)

    P = np.zeros(x.shape)
    for j in range(n+1):
        P = P + res[j] * x**(2*j)
    return res, P

"____________________________________________________________________________"

def least_squares_Q(x, y, n):
    m = x.size

    N = 2*n+1
    b_start = n+1
    A = np.zeros((m, N))
    for j in range(n+1):
        A[:, j] = x**j
    for j in range(1, n+1):
        A[:, j+b_start-1] = -y * x**j

    params = least_squares(A, y)
    Q = calc_Q(x, params)
    return params, Q

"____________________________________________________________________________"

def calc_Q(omega, params):

    # parameters
    N = len(params)
    n = int((N-1)/2)
    a = params[:n+1]
    b = np.array([0, *params[n+1:]])

    #Temporary variables
    num = np.zeros(omega.shape)
    den = np.zeros(omega.shape)
    for i in range(n+1):
        num = num + a[i] * omega**i
        den = den + b[i] * omega**i
    result = num/(1 + den)
    return result



"____________________________________________________________________________"

print("\nAnswer a2:")
for i in omega:
	M = E-i*S
	Cond_num = cond(M)
	print(f"Condition number for each omega = {i}")
	print(Cond_num)
	print(np.floor(-np.log10(Cond_num*z)))

"____________________________________________________________________________"


print("\nAnswer b1:")
for i in omega:
	e = error_bound(E, S, i)
	print(f"Error bound for omega={i}:")
	print(e)
	print(np.floor(-np.log10(e)))
"____________________________________________________________________________"


M = np.array([[2, 1, 1], [4, 1, 4], [-6, -5., 3]])
b = np.array([4, 11, 4.])

print("\nAnser c1:")
if lu_solve(M,b).all() == np.linalg.solve(M, b).all():
	print("it worked !!! I knew you had it in you")
	print(lu_solve(M,b))
	print(np.linalg.solve(M, b))
else:
	print("I have a bad feeling about this, something is wrong")

"____________________________________________________________________________"


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

"____________________________________________________________________________"

#Week 2 

#F1
#Check that QR decomposition works
"____________________________________________________________________________"

print("\n Answer to F")
Q, R = qr_factorize(A2)

identity_check = Q.T@Q
R_check = Q@R

print(identity_check)
print("\n")
print(R_check)
print(f'Matrix Q = \n {Q}')
print(f'Matrix R = \n {R}') 
"____________________________________________________________________________"

R, vec = householder_fast(A2)

print(f'Matrix R = \n {R}') 

"____________________________________________________________________________"


#F3
x = least_squares(A2,b2)

print("\n Slow Householder QR decomposition on matrix A2")
print(f'Resulting R matrix: R= \n {R}')
print("\n Least Squares method on matrix A2 and vector b2")
print(f'linear least square fit: x={x}')

"____________________________________________________________________________"


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

print("g parameters")
print(x_1)
print(x_2)

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

a1 = params_1[:3]
b1 = params_1[3:]
a2 = params_2[:5]
b2 = params_2[5:]
print("h: parameters")
print(a1, b1)
print(a2, b2)

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

