from fileinput import close
from symbol import power
from winreg import KEY_ENUMERATE_SUB_KEYS
import numpy as np
import matplotlib.pyplot as plt
from LU_routines import least_squares
from examplematrices import *
from chladni_show import *

### Linea S. Hedemark ###
###       qrg977      ###

### a ###

##a1 gershgorin

def gershgorin(A):
    centers = np.diag(A)
    radii = np.sum(np.abs(A),axis = 1) - np.abs(centers)
    return centers, radii

test_m = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(test_m, gershgorin(test_m))

###a2 use gershgorin routine on Kmat
#Kmat = np.load('OneDrive - University of Copenhagen/FysiskeFag/Master/SciComp/project2/Chladni-Kmat.npy')
Kmat = np.load('Chladni-Kmat.npy')
Kcenters, Kradii = gershgorin(Kmat)
print("results for a2")
for i in range(len(Kcenters)):
    print(i, ": ", Kcenters[i], Kradii[i])

#this is just a plot with drawings of the circles
fig, ax = plt.subplots()
ax. plot(Kcenters, np.zeros(len(Kcenters)), '.')
for i in range(len(Kcenters)):
    circle = plt.Circle( (Kcenters[i],0), Kradii[i], fill = False) 
    ax.add_artist(circle)
ax.set_ylim(-100000, 100000)
ax.set_xlim(-6000, 180000)
plt.show()


### b ###

#b1: rayleigh quotient
def rayleigh_qt(A, x):
    lamb = np.dot(x, np.dot(A,x))/np.dot(x,x)
    return lamb

#b2: power iterate

def power_iterate(A, x0, max_iter = 100, conv_limit = 10**(-10)):
    #normalised, with max-norm
    x_i = x0.copy()

    #as a convergence measure, we check the eigenvalue, and break if the changes are small 
    # enough that we deem it is converging
    lambda_before = rayleigh_qt(A, x_i)
    for i in range(max_iter):
        #next vector
        y_i = np.dot(A, x_i)

        #normalise
        x_i = y_i/np.linalg.norm(y_i, np.inf)

        k = i+1
        #calculate new approx. eigenvalue
        lambda_new = rayleigh_qt(A, x_i)
        ratio = abs((lambda_new-lambda_before)/lambda_before)
        if ratio <= conv_limit:
            break 
        lambda_before = lambda_new

    return x_i, k

#b3: find eigenvalue, Rayleigh residual and number of iterations for example matrices

def Bresult(A):
    #random vector to start process
    start_vec = np.random.random(size = len(A))

    #find eigenvector and number of iterations
    x, niter = power_iterate(A, start_vec)

    #calculate approximate eigenvalue
    eigval = rayleigh_qt(A, x)

    #calculate Rayleigh residual
    ralres = np.sqrt(np.sum((np.dot(A, x)-eigval*x)**2))
    return eigval, ralres, niter

test_matrices = [A1, A2, A3, A4, A5, A6]
#print results
print("b3 results:")
for i in range(len(test_matrices)):
    print("Result for matrix A", i+1, " is ", Bresult(test_matrices[i]))

#b4
#initialise random vector for power iteration
K_init_vec = np.random.random(len(Kmat))
K_eigfunc = power_iterate(Kmat, K_init_vec)[0]
print("Biggest eigenvalue (approximate) for K is: ", rayleigh_qt(Kmat, K_eigfunc))
show_waves(K_eigfunc)
show_nodes(K_eigfunc)


def rayleigh_iterate(A, x0, shift0, epsilon = 10**(-8),  max_iter = 100):
    n = A.shape[0]
    xi = x0.copy()
    shift = shift0

    #perform a couple of steps of shifted inverse to ensure x is good eigenvector
    for i in range(5):
        matrix = A - shift * np.identity(n)
        y = least_squares(matrix, xi)[0]
        xi = y/np.linalg.norm(y,np.inf)

    for i in range(max_iter):
        matrix = A - shift*np.identity(n)
        y = least_squares(matrix, xi)[0]

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


def Cresult(A):
    #random vector to start process
    np.random.seed(1)

    start_vec = np.random.random(size = len(A))

    #find eigenvector and number of iterations
    x, niter = rayleigh_iterate(A, start_vec, 0)

    #calculate approximate eigenvalue
    eigval = rayleigh_qt(A, x)

    #calculate Rayleigh residual
    ralres = np.sqrt(np.sum((np.dot(A, x)-eigval*x)**2))
    return eigval, ralres, niter

#print results
print("c2 results:")
for i in range(len(test_matrices)):
    print("Result for matrix A", i+1, " is ", Cresult(test_matrices[i]))

#print(rayleigh_iterate(A3, np.random.random(len(A3)), 0))
#print(rayleigh_qt(A3, np.array([ 1.        , -0.82894514, -0.10020662])))

### d ###

##as I don't trust my own QR routine as it's giving problems, I'm defining a new rayleigh
# iterate for this purpose using Numpy's routine for solving matrix

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

#First, check how many unique eigenvalues there are in K, so that we know how many to look for
print("Number of unique eigenvalues in K: ", len(np.unique(np.linalg.eig(Kmat)[0])))

#We use the gershgorin circles information to inform our shift guesses
# We have the centres, and then the centre "edges", so plus/minus the radius

center_eig = []
left_eig = []
right_eig = []

for i in range(len(Kcenters)):
    x_c, e_c, k_c, res_c = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=Kcenters[i])
    x_l, e_l, k_l, res_l = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=(Kcenters[i]-Kradii[i]))
    x_r, e_r, k_r, res_r = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=(Kcenters[i]+Kradii[i]))

    center_eig.append(e_c)
    left_eig.append(e_l)
    right_eig.append(e_r)

def find_unique(eigs):#, rerr = 1e-05, aerr = 1e-08):
    #where will we save the unique eigs
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

all_found_eigs = np.concatenate((center_eig, left_eig, right_eig))
distinct_eigs = find_unique(all_found_eigs)
print("Have we found all the eigenvalues?")
print("The number of distinct eigenvalues found are: ", len(distinct_eigs))

print("Smallest eigenvalue: ", np.min(distinct_eigs))

# Check that the lowest eigenfunction looks like a cross
x_low, eigval_low, k_low, resi_low = np_rayleigh_iterate(Kmat, x0=np.random.rand(Kmat.shape[0]), shift0=13)

# visualise the eigenfunction
show_nodes(x_low, basis_set)

#d3 Construct T matrix using all the eigenvectors

#could I have done this a smarter way and combined everything in one loop?
#yes
#did?
#no

#to construct T we need the 15 eigenvalues and corresponding functions

#somewhere to store the eigen-things
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


### the following plotting magic is graciously provided by Kimi Kreilgaard

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

#d4 visualise all nodes
show_all_wavefunction_nodes(T,np.array(eigval),basis_set)