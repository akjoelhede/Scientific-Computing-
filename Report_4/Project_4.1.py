import numpy as np
import matplotlib.pyplot as plt


#A CLASS DOES NOT TAKE ANY PARAMETERS OR VALUES
class HIV():

	def __init__(self, Var, parameters ): # THE __INIT__ FUNCTION DEFINES THE PARAMETERS IN THE CLASS
		self.parameters = parameters
		self.Var = Var

		
	#ALL FUNCTIONS BELOW THIS ARE CALLABLE 

	def Diff(self, Var, parameters):

		x1, x2, y, z = Var
		
		a1, a2, b1, b2, b3, c1, c2, d1, p1, p2, r, q, e, r1, r2, r3, r4 = parameters

		diffx1 = a1*x1*(p1-x1)+ a2*x2*(p1-x1) + e*(p1-x1) - r1*x1

		diffx2 = b1*x1*(p2-x2)+b2*x2*(p2-x2) + b3*y*(p2-x2) + e*(p2-x2)- r2*x2

		diffy = c1*x2*(q-y) + c2*z*(q-y) + e*(q-y) - r3*y

		diffz = d1*y*(r-z) + e*(r-z) - r4*z

		next = np.array([diffx1, diffx2, diffy, diffz])

		return diffx1, diffx2, diffy, diffz

	
	def euler(self, Var, diff, dt):

		euler_step = Var + diff * dt

		return euler_step

	def runge_kutta(self, Var, parameters, dt):

		k1 = Diff(self, Var, parameters)
		k2 = Diff(self, Var + (dt/2)*k1, parameters)
		k3 = Diff(self, Var + (dt)*k2, parameters)
		k4 = Diff(self, Var + dt*k3, parameters)

		new_X = Var + (dt/ 6 ) * (k1 + 2*k2 + 2*k3  +k4)

		return new_X

dt = 1

x1 = 0.01
x2 = 0
y = 0
z = 0

a1 = 10
a2 = 5
b1 = 5
b2 = 0
b3 = 0
c1 = 0
c2 = 0
d1 = 0
p1 = 5
p2 = 5
r = 100 
q = 100 
e = 0
r1 = 0
r2 = 0
r3 = 0 
r4 = 0

Var = np.array([x1, x2, y, z])

parameters = np.array([a1, a2, b1, b2, b3, c1, c2, d1, p1, p2, r, q, e, r1, r2, r3, r4])

#THIS CALLS THE CLASS
s = HIV(Var = Var, parameters=parameters)

#CALL TO DIFF FUNCTION
diff = s.Diff(Var, parameters)

euler_store = []
#CALL TO EULER FUNCITON
for i in range(100):
	euler = s.euler(Var, diff, dt)
	euler_store.append(euler)
print(euler_store)
#CALL TO RUNGE-KUTTA FUNTION
Runge_Kutta = s.runge_kutta(Var, parameters, dt)





