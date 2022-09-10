import numpy as np

def max_norm(M):
	norm = np.max(np.sum(np.abs(M), axis=1))
	return norm

def cond(M):
	M_inv = np.linalg.inv(M)
	M_norm = max_norm(M)
	M_inv_norm = max_norm(M)
	condition_num = M_norm * M_inv_norm
	return condition_num

def error_bound(E, S, omega):
	M = E-omega*S
	Cond_num = cond(M)
	Max_norm = max_norm(M)

	return Cond_num * max_norm(S*1/2*10**(-3))/ Max_norm

def lu_factorize(M):
    
    #Get the number of rows
	n = M.shape[0]
    
	U = M.copy()
	L = np.eye(n)
    
    #Loop over rows
	for i in range(n):
            
        #Eliminate entries below i with row operations 
        #on U and reverse the row operations to 
        #manipulate L
		factor = U[i+1:, i] / U[i, i]
		L[i+1:, i] = factor
		U[i+1:] -= factor[:, np.newaxis] * U[i]
	return L, U

def forward_substitution(L, b):
    
	y = np.zeros(b.shape)
	for i in range(b.size):
		y[i] = b[i] - L[i, :i].dot(y[:i])
	return y


def back_substitution(U, y):
    
	x = np.zeros(y.shape)
	for i in reversed(range(y.size)):
		x[i] = (y[i] - U[i, i:].dot(x[i:]))/U[i, i]
	return x

def lu_solve(M, b):
    
	L, U = lu_factorize(M)
    
	y = forward_substitution(L, b)
    
	return back_substitution(U, y)

def solve_alpha(omega, E, S, z):
	
	M = E-omega*S

	L,U = lu_factorize(M)
	y = forward_substitution(L,z)
	x = back_substitution(U,y)
	alpha = z.dot(x)

	return alpha 