
#I learned from my friend Nikolai Nielsen how to import from other variables and functions from other files 
# and i figured that i would use that to since the code will be much nicer to look at
from watermatrices import Amat, Bmat, yvec, Imat
from subroutines import *
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

 