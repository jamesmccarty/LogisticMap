#!/usr/bin/env python

import matplotlib.pyplot as plt
from numpy import *

''' plot a bifurcation diagram of the logistic map ''' 

# define logistic map function
def LogisticMap(r,x):
   return r*x*(1.0-x)

seed=0.25   # set seed
nskip=100   # how many iterations to skip before plotting
npoints=500  # how many iterations to plot

# This determines the increment of r values
rinc=(4.0-2.0)/float(npoints)

# Now we set up the figure for plotting
plt.figure(1,(8,6))
plt.xlabel('r')
plt.ylabel('x_i')

plt.plot([4.0],[1.0],'k,')
plt.plot([4.0],[0.0],'k,')
plt.plot([2.0],[0.0],'k,')
plt.plot([2.0],[1.0],'k,')

# Iterate the logistic map
for r in arange(2,4,rinc):
   X0=seed
   ### Initial iteration before plotting
   Ninit=nskip
   for j in xrange(Ninit):
      X0= LogisticMap(r,X0)
   ## now iterate npoints with plotting
   rsweep = []  # array of r values
   x = []    # array of iterates
   for k in xrange(npoints):
      X0=LogisticMap(r,X0)
      rsweep.append(r)
      x.append(X0)

   plt.plot(rsweep,x,'k,')   # Plot the list of (r,x) pairs

plt.show()
