#Anders Terp Kjoelhede
#Date: 21/10/2022
#Scientific Computing
#Project_4
#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
from numba import njit

#%%
def sink(matrix):

	matrix = matrix.copy()

	matrix[0,1:-1] = matrix[1, 1:-1]

	matrix[-1,1:-1] = matrix[-2,1:-1]

	matrix[1:-1,0] = matrix[1:-1,1]

	matrix[1:-1,-1] = matrix[1:-1,-2]

	return matrix

@njit
def laplace(matrix, step_size = 1):

	inner = matrix[1:-1, 1:-1]
	up = matrix[:-2,1:-1]
	down = matrix[2:,1:-1]
	left = matrix[1:-1,:-2]
	right = matrix[1:-1,2:]

	diffmatx = (1/step_size**2) * (left -2*inner + right)

	diffmaty = (1/step_size**2) * (down -2*inner + up)

	Lap = diffmatx + diffmaty

	return Lap

D_p = 1
D_q = 8
C = 4.5
K = 9

@njit
def update(Pmatrix, Qmatrix, dt, step_size=1, parameters=(D_p, D_q, C, K)):

	#Extract parameters from outside function
	D_p, D_q, C, K = parameters[0], parameters[1], parameters[2], parameters[3]

	#Get inner matrix for both P and Q
	Pinner = Pmatrix[1:-1, 1:-1]
	Qinner = Qmatrix[1:-1, 1:-1]

	#Get laplacians
	Lap_P = laplace(Pmatrix, step_size)
	Lap_Q = laplace(Qmatrix, step_size)

	#Define coupled differential equations
	diff_Pmatrix = D_p * Lap_P + Pinner**2 * Qinner + C - (K+1)*Pinner
	diff_Qmatrix = D_q * Lap_Q - Pinner**2 * Qinner + K*Pinner

	#Calculate new matrix
	New_P = Pinner + dt * diff_Pmatrix
	New_Q = Qinner + dt * diff_Qmatrix

	#Rescale mactrices to original size
	Pmatrix[1:-1, 1:-1] = New_P
	Qmatrix[1:-1, 1:-1] = New_Q

	return Pmatrix, Qmatrix


def plot_function(P,Q):
	fig, ax = plt.subplots(ncols=2)
	mini, maks = np.min((P,Q)), np.max((P,Q))
	im = ax[0].imshow(P, cmap = 'Greys', vmin = mini, vmax = maks)
	ax[0].set_title('P with ghosts')
	ax[1].imshow(Q, cmap='Greys', vmin = mini, vmax=maks)
	ax[1].set_title('Q with ghosts')
	cax = plt.axes([0.95, 0.2, 0.05, 0.58])
	fig.colorbar(im, cax=cax)
	plt.show()

dx = 0.5

#Length of side of matrix
N = int(43/dx)

P_mat = np.zeros((N,N))
Q_mat = np.zeros((N,N))

low, high = int(12/dx), int(31/dx)
P_mat[low:high,low:high] = C + 0.1
Q_mat[low:high,low:high] = K/C + 0.2

dt = 0.001
t = 2000

P_frames = []
Q_frames = []

t_freeze = [0, 5e2, 1e3, 1.5e3, 2e3, 3e3,
	4e3, 5e3, 1e4, 2.5e4, 5e4, 1e5,
	2e5, 3e5, 5e5, 7.5e5, 1e6, 2e6-1]

for time in tqdm(range(int(t/dt))):

	P_mat = sink(P_mat)
	Q_mat = sink(Q_mat)

	if time in t_freeze:

		P_frames.append(P_mat)
		Q_frames.append(Q_mat)

	P_mat, Q_mat = update(P_mat, Q_mat, dt, parameters = (D_p, D_q, C, K), step_size=dx)

#%%

plot_function(P_frames[-1], Q_frames[-1])

#%%
anim_50 = Q_frames

fig = plt.figure(figsize = (8,8))
im = plt.imshow(anim_50[0])
plt.colorbar()
def animate_func(i):
	im.set_array(anim_50[i])
	return [im]

anim = animation.FuncAnimation(fig, animate_func, frames = len(P_frames), interval = 20)

anim.save('Q_matrix.mp4', fps = 2) 

plt.show()
plt.close('All')
# %%

D_p, D_q, C = 1, 8, 4.5
K_arr = np.arange(7,13,1)

P_list = []
Q_list = []

dx = 0.5

N = int( 43 / dx )

for K in tqdm(K_arr):

	P_mat = np.zeros((N,N))
	Q_mat = np.zeros((N,N))

	low, high = int(12/dx), int(31/dx)
	P_mat[low:high,low:high] = C + 0.1
	Q_mat[low:high,low:high] = K/C + 0.2

	dt = 0.001
	t = 2000

	for i in range( int(t/dt) ):
        
		P_mat = sink(P_mat)
		Q_mat = sink(Q_mat)

		P_mat, Q_mat = update(P_mat, Q_mat, dt, parameters=(D_p, D_q, C, K), step_size=dx)

		P_list.append(P_mat)
		Q_list.append(Q_mat)

fig, ax = plt.subplots(nrows=2, ncols=len(K_arr), figsize=(16,6), gridspec_kw={'wspace':0.2, 'hspace':0.03})

mini, maks = np.min((P_list, Q_list)), np.max((P_list, Q_list))

for i in range(len(K_arr)):
	im = ax[0,i].imshow(P_list[i], cmap='plasma', vmin=mini, vmax=maks)
	ax[0,i].set_title('P (K={}, t=2000)'.format(K_arr[i]))
	ax[0,i].get_xaxis().set_visible(False)
	ax[0,i].get_yaxis().set_visible(False)

	ax[1,i].imshow(Q_list[i], cmap='plasma', vmin=mini, vmax=maks)
	ax[1,i].set_title('Q (K={}, t=2000)'.format(K_arr[i]))
	ax[1,i].get_xaxis().set_visible(False)
	ax[1,i].get_yaxis().set_visible(False)


cax = plt.axes([0.92, 0.15, 0.02, 0.7]) #[left, bottom, width, height]
fig.colorbar(im, cax=cax)


plt.savefig('k_all.png')
plt.show()
# %%
