import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


#A CLASS DOES NOT TAKE ANY PARAMETERS OR VALUES
class HIV():

	def __init__(self, parameters ): # THE __INIT__ FUNCTION DEFINES THE PARAMETERS IN THE CLASS
		self.parameters = parameters

		
	#ALL FUNCTIONS BELOW THIS ARE CALLABLE 

	def Diff(self,parameters):
		

		x1_new = parameters[0]*x1*(p1-x1)+ a2*x2*(p1-x1)

		x2_new = b1*x1*(p2-x2)+b2*x2*(p2-x2)+ b3*y*(p2-x2)

		y_new = c1*x2*(q-y) + c2*z*(q-y)

		z_new = d1*y*(r-z)

		return x1_new, x2_new, y_new, z_new


parameters = np.array([a1, a2, b1, b2, b3, c1, c2, d1, x1, x2, p1, p2, y, r, z])

x = np.linspace(2,6)
y = np.linspace(3,7)

#THIS CALLS THE CLASS AND ASSIGNS IT TO A VARIABLE S. X AND Y ARE GIVEN AS PARAMETERS
s = Twoparticles(x=x, y = y)

#HERE I CALL A FUNCTION WITHIN THE CLASS TO CALCULATE THE POTENTIAL ENERGY
pot = s.pot_run(x, y)

#HERE I CALL ANOTHER FUNCTION WITHIN THE CLASS EASILY
kin = s.kin_run(x, y)

print(pot, kin)

