import numpy as np
import matplotlib.pyplot as plt


#A CLASS DOES NOT TAKE ANY PARAMETERS OR VALUES
class HIV():

	def __init__(self,parameters ): # THE __INIT__ FUNCTION DEFINES THE PARAMETERS IN THE CLASS
		self.parameters = parameters

		
	#ALL FUNCTIONS BELOW THIS ARE CALLABLE 

	def Diff(self,parameters):
		
		x1, x2, y, z, a1, a2, b1, b2, b3, c1, c2, d1, p1, p2, r, q = parameters

		diffx1 = a1*x1*(p1-x1)+ a2*x2*(p1-x1)

		diffx2 = b1*x1*(p2-x2)+b2*x2*(p2-x2) + b3*y*(p2-x2)

		diffy = c1*x2*(q-y) + c2*z*(q-y)

		diffz = d1*y*(r-z)

		return diffx1, diffx2, diffy, diffz

	
	def euler(self, x0, diff, dt):

		x_new = x0 + diff * dt

		return x_new
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

parameters = np.array([x1, x2, y, z, a1, a2, b1, b2, b3, c1, c2, d1, p1, p2, r, q])

#THIS CALLS THE CLASS AND ASSIGNS IT TO A VARIABLE S. X AND Y ARE GIVEN AS PARAMETERS
s = HIV(parameters=parameters)

#HERE I CALL A FUNCTION WITHIN THE CLASS TO CALCULATE THE POTENTIAL ENERGY
diffx1, diffx2, diffy, diffz = s.Diff(parameters)

#HERE I CALL ANOTHER FUNCTION WITHIN THE CLASS EASILY
new_step = s.euler(x1, diffx1, 0.1)
print(new_step)




