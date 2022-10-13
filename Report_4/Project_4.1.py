import numpy as np
import matplotlib.pyplot as plt


#A CLASS DOES NOT TAKE ANY PARAMETERS OR VALUES
class HIV():

	def __init__(self,x1, x2, y, z,parameters ): # THE __INIT__ FUNCTION DEFINES THE PARAMETERS IN THE CLASS
		self.parameters = parameters
		self.x1 = x1
		self.x2 = x2
		self.y = y
		self.z = z

		
	#ALL FUNCTIONS BELOW THIS ARE CALLABLE 

	def Diff(self,x1, x2, y, z, parameters):
		

		x1_new = parameters[0]*x1*(parameters[10]-x1)+ parameters[1]*x2*(parameters[10]-x1)

		x2_new = parameters[2]*x1*(parameters[11]-x2)+parameters[3]*x2*(parameters[11]-x2) + parameters[4]*y*(parameters[11]-x2)

		y_new = parameters[5]*x2*(parameters[15]-y) + parameters[6]*z*(parameters[15]-y)

		z_new = parameters[7]*y*(parameters[13]-z)

		return x1_new, x2_new, y_new, z_new


			0   1   2   3   4   5   6   7   8  9  10  11 
parameters = np.array([a1, a2, b1, b2, b3, c1, c2, d1, p1, p2, r, q])

#THIS CALLS THE CLASS AND ASSIGNS IT TO A VARIABLE S. X AND Y ARE GIVEN AS PARAMETERS
s = HIV(parameters=parameters)

#HERE I CALL A FUNCTION WITHIN THE CLASS TO CALCULATE THE POTENTIAL ENERGY
pot = s.Diff(parameters)

#HERE I CALL ANOTHER FUNCTION WITHIN THE CLASS EASILY
kin = s.Euler(parameters)



