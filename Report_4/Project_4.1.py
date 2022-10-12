import numpy as np
import matplotlib.pyplot as plt


#A CLASS DOES NOT TAKE ANY PARAMETERS OR VALUES
class HIV():

	def __init__(self, parameters ): # THE __INIT__ FUNCTION DEFINES THE PARAMETERS IN THE CLASS
		self.parameters = parameters

		
	#ALL FUNCTIONS BELOW THIS ARE CALLABLE 

	def Diff(self,parameters):
		

		x1_new = parameters[0]*parameters[8]*(parameters[10]-parameters[8])+ parameters[1]*parameters[9]*(parameters[10]-parameters[8])

		x2_new = parameters[2]*parameters[8]*(parameters[11]-parameters[9])+parameters[3]*parameters[9]*(parameters[11]-parameters[9])+ parameters[4]*parameters[12]*(parameters[11]-parameters[9])

		y_new = parameters[5]*parameters[9]*(parameters[15]-parameters[12]) + parameters[6]*parameters[14]*(parameters[15]-parameters[12])

		z_new = parameters[7]*parameters[12]*(parameters[13]-parameters[14])

		return x1_new, x2_new, y_new, z_new


	def Euler(self, parameters, ):


parameters = np.array([a1, a2, b1, b2, b3, c1, c2, d1, x1, x2, p1, p2, y, r, z, q])

#THIS CALLS THE CLASS AND ASSIGNS IT TO A VARIABLE S. X AND Y ARE GIVEN AS PARAMETERS
s = HIV(parameters=parameters)

#HERE I CALL A FUNCTION WITHIN THE CLASS TO CALCULATE THE POTENTIAL ENERGY
pot = s.Diff(parameters)

#HERE I CALL ANOTHER FUNCTION WITHIN THE CLASS EASILY
kin = s.kin_run(parameters)

print(pot, kin)

