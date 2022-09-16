import numpy as np

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

"____________________________________________________________________________"

def least_squares_Q(x, y, n):
    """
    Performs a linear least squares fit to a rational approximation function -
    using a linear approximation for Q:
    Q = a_j omega^j - Q b_j omega^j, and Q \approx alpha
    So we perform the linear fit on the system
    alpha = a_j omega^j - alpha b_j omega^j
    And then substitute these parameters into Q to use for approximating alpha.
    """
    m = x.size

    # There are 2n+1 parameters: a_j for j=0,...,n and b_j for j=1,...,n
    N = 2*n+1
    b_start = n+1
    # Build the matrix needed
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
    """
    Rational approximating function of the form
    Q = [sum(a_j omega^j, 0, n)] / [1 + sum(b_j omega^j, 1, n)]
    """

    # Split the parameters
    N = len(params)
    n = int((N-1)/2)
    a = params[:n+1]
    # Add a zero as the first parameter of b, so only one for loop is needed.
    b = np.array([0, *params[n+1:]])

    # Create temp variables for the results.
    num = np.zeros(omega.shape)
    den = np.zeros(omega.shape)
    for i in range(n+1):
        num = num + a[i] * omega**i
        den = den + b[i] * omega**i
    result = num/(1 + den)
    return result