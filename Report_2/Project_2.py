from curses.ascii import BEL
import numpy as np
import matplotlib.pyplot as plt
from example_matrices import *
from chladni_show import *

"Anders Terp Kjølhede, nzd737"




Kmat = np.load("Chladni-Kmat.npy")

Test_matrix = np.array([[1,2,3],[4,6,3],[7,4,8]])
test_vector = np.array([1,2,3])

"a1_______________________________________________________________"
def gershgorin(A):
	centers = np.diag(A)
	radius = np.sum(np.abs(A), axis=1)-np.abs(centers) 
	return centers, radius

"a2_______________________________________________________________"

centers, radius = gershgorin(Kmat)

print(centers, radius)

"b1________________________________________________________________"
def rayleigh_qt(A,x):
	lamb = (x.T).dot(A@x)/((x.T).dot(x))
	return lamb


print(rayleigh_qt(Test_matrix,test_vector))

"b2________________________________________________________________"

def power_iterate(A, x0=None, max_iter=25, conv_criterion=1e-6):
    
    n, _ = A.shape
    # Choose random x0 if not given one
    x = np.random.uniform(size=n) if x0 is None else x0

    lambda_last = rayleigh_qt(A,x)

    for i in range(max_iter):
        y = A@x
        x = y / np.amax(y)

        lambda_new = rayleigh_qt(A,x)
        res = abs((lambda_new-lambda_last)/(lambda_last))
        if res <= conv_criterion:
            break
        lambda_last = lambda_new

    return x, i+1

print(power_iterate(Test_matrix,test_vector))

"b3________________________________________________________________"


x, k = power_iterate(A1)

approx = rayleigh_qt(A1,x)

res = np.sqrt(np.sum((A1@x-approx*x)**2))

print(f'A1 converged after {k} iterations')

print(f"Eigenvalue: {approx}")
print(f'Residual: {res}')

def Bresult(A):

    #find eigenvector and number of iterations
    x, k = power_iterate(A)

    #calculate approximate eigenvalue
    approx = rayleigh_qt(A,x)

    #calculate Rayleigh residual
    res = np.sqrt(np.sum((A@x-approx*x)**2))
    return approx, res, k

test_matrices = [A1, A2, A3, A4, A5, A6]

#print results
print("b3 results:")
for i in range(len(test_matrices)):
    print("Result for matrix A", i+1, " is ", Bresult(test_matrices[i]))


"b4________________________________________________________________"

x, k = power_iterate(Kmat, conv_criterion=1e-9, max_iter=50)
eig = rayleigh_qt(Kmat, x)
res = np.sqrt(np.sum((Kmat@x - eig*x)**2))

print(f'K converged after {k} iterations')

print(f'eigenvalue: {eig}')
print(f'Residual: {res}')

show_nodes(x)

show_waves(x)

"c1________________________________________________________________"
def back_substitution(U, y):
    
	x = np.zeros(y.shape)
	for i in reversed(range(y.size)):
		x[i] = (y[i] - U[i, i:].dot(x[i:]))/U[i, i]
	return x

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

def rayleigh_iterate(A, x0, shift0, epsilon = 10**(-8),  max_iter = 100):

    n = A.shape[0]
    xi = x0.copy()
    shift = shift0

    #perform a couple of steps of shifted inverse to ensure x is good eigenvector
    for i in range(5):
        matrix = A - shift * np.identity(n)
        y = least_squares(matrix, xi)
        xi = y/np.linalg.norm(y,np.inf)

    for i in range(max_iter):
        matrix = A - shift*np.identity(n)
        y = least_squares(matrix, xi)

        #normalise
        xi = y/np.linalg.norm(y,np.inf)

        #count how many iterations we needed so far
        k = i+1

        #new eigenvalue approximation
        shift = rayleigh_qt(A, xi)

        #compute rayleigh residual as a convergence measure
        rayleigh_res = np.sqrt(np.sum((np.dot(A, xi)-shift*xi)**2))

        if rayleigh_res <= epsilon:
            break
    return xi, k


"c2________________________________________________________________"

def Cresult(A):

    np.random.seed(42)

    start_vec = np.random.random(size = len(A))

    #find eigenvector and number of iterations
    x, k = rayleigh_iterate(A,start_vec, 0)

    #calculate approximate eigenvalue
    approx = rayleigh_qt(A, x)

    #calculate Rayleigh residual
    res = np.sqrt(np.sum((np.dot(A, x)-approx*x)**2))
    return approx, res, k

#print results
print("c2 results:")
for i in range(len(test_matrices)):
    print("Result for matrix A", i+1, " is ", Cresult(test_matrices[i]))


"d2________________________________________________________________"

#Because my own QR factorization were too unstable around singular matrices i choose to use np.solve instead

def np_rayleigh_iterate(A, x0, shift0, epsilon = 10**(-10), max_iter = 100):
    n = A.shape[0]
    xi = x0.copy()
    shift = shift0

    #perform a couple of steps of shifted inverse to ensure x is good eigenvector
    for i in range(5):
        matrix = A - shift * np.identity(n)
        y = np.linalg.solve(matrix, xi)
        xi = y/np.linalg.norm(y,np.inf)

    for i in range(max_iter):
        matrix = A - shift*np.identity(n)
        y = np.linalg.solve(matrix, xi)

        #normalise
        xi = y/np.linalg.norm(y,np.inf)

        #count how many iterations we needed so far
        k = i+1

        #new eigenvalue approximation
        shift = rayleigh_qt(A, xi)

        #compute rayleigh residual as a convergence measure
        rayleigh_res = np.sqrt(np.sum((np.dot(A, xi)-shift*xi)**2))

        if rayleigh_res <= epsilon:
            break

    return xi, shift, k, rayleigh_res


# Firstly we will check how many eigenvalues in K we are looking for
print("Number of unique eigenvalues in K: ", len(np.unique(np.linalg.eig(Kmat)[0])))


#Generate a lot approximate eigenvalues 
center_eig = []
left_eig = []
right_eig = []

for i in range(len(centers)):
    x_c, e_c, k_c, res_c = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=centers[i])
    x_l, e_l, k_l, res_l = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=(centers[i]-radius[i]))
    x_r, e_r, k_r, res_r = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=(centers[i]+radius[i]))

    center_eig.append(e_c)
    left_eig.append(e_l)
    right_eig.append(e_r)

#Find the unique eigenvalues 
def find_unique(eigs):#, rerr = 1e-05, aerr = 1e-08):
    unique_eigs = []

    while len(eigs) > 0:
        #compare first element in eigenlist with remaining list and use np.isclose to determine 
        # whether they are close
        close_bud = np.isclose(eigs[0], eigs)

        #take the mean of those values that are close
        mean_buddies = np.mean(eigs[close_bud])

        unique_eigs.append(mean_buddies)

        eigs = eigs[~close_bud]

    return np.sort(unique_eigs)

#Collect all eigenvalues in list and then find all the unique ones
all_found_eigs = np.concatenate((center_eig, left_eig, right_eig))
distinct_eigs = find_unique(all_found_eigs)
print("The number of distinct eigenvalues found are: ", len(distinct_eigs))

#Fnd the smallest eigenvalue
print("Smallest eigenvalue: ", np.min(distinct_eigs))

#Check that the lowest eigenvalue looks like a cross
x_low, eigval_low, k_low, resi_low = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=13)

# visualise the eigenfunction
show_nodes(x_low, basis_set)

"d3________________________________________________________________"
eigval = []
eigvec = []
for i in range(15):
    #change them a lil so rayleigh doesn't know we know
    shifts = distinct_eigs[i] + 0.3

    x_, eigval_, k_, resi_ = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=shifts)

    eigval.append(eigval_)
    eigvec.append(x_)

#the eigenvalues are already sorted from lowest to highest

T = np.zeros_like(Kmat)
for i in range(15):
    T[:,i] = eigvec[i]

La = np.diag(np.array(eigval))


### Provided by Kimi Kreilgaard, THANKS

# Visualize it
fig, ax = plt.subplots(ncols=3, figsize=(15,5), gridspec_kw={'wspace':0.1})

# LHS
mini = np.min([Kmat,T @ La @ np.linalg.inv(T)]) 
maxi = np.max([Kmat,T @ La @ np.linalg.inv(T)])
im = ax[0].imshow(Kmat, vmin=mini, vmax=maxi, cmap='Blues')
ax[0].set_title(r'$\mathbf{K}$')

# RHS
ax[1].imshow(T @ La @ np.linalg.inv(T), vmin=mini, vmax=maxi, cmap='Blues')
ax[1].set_title(r'$\mathbf{T}\Lambda\mathbf{T^{-1}}$')

# LHS - RHS
ax[2].imshow(Kmat - T @ La @ np.linalg.inv(T), vmin=mini, vmax=maxi, cmap='Blues')
ax[2].set_title(r'$\mathbf{K}-\mathbf{T}\Lambda\mathbf{T^{-1}}$')

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()

"d4________________________________________________________________"
# visualise all nodes
show_all_wavefunction_nodes(T,np.array(eigval),basis_set)



