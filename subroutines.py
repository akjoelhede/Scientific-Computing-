import numpy as np

def max_norm(M):
	norm = np.max(np.sum(np.abs(M), axis=1))
	return norm

def cond(M):
	M_inv = np.linalg.inv(M)
	M_norm = max_norm(M)
	M_inv_norm = max_norm(M_inv)
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
    
    #Loop over the rows 
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

def make_householder(a):
    #find prependicular vector to mirror
    u = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    u[0] = 1
    H = np.eye(a.shape[0])
    #finding Householder projection
    H -= (2 / np.dot(u, u)) * np.outer(u,u)
    return H

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
 
def least_squares(A, b):
    m, n = A.shape
    Q, R = qr_factorize(A)
    b2 = Q.T@b
    x = back_substitution(R[:n], b2[:n])

    return x

def least_squares_P(x, y, n):
    # We run the sum from j=0 to n, so we have n+1 terms, and n+1 parameters
    # (and n+1 columns in our matrix)
    m = x.size

    # Now we just create our matrix and solve the least squares problem
    A = np.zeros((m, n+1))
    for j in range(n+1):
        A[:, j] = x**(2*j)
    res = least_squares(A, y)

    P = np.zeros(x.shape)
    for j in range(n+1):
        P = P + res[j] * x**(2*j)
    return res, P
