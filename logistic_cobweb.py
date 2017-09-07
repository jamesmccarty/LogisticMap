#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt

''' cobweb and iterate plot of the logistic map ''' 

# define logistic map function
def LogisticMap(r,x):
   return r*x*(1.0-x)

def plot_cobweb(x0,r,n):
    xval = np.linspace(0.0,1.0,2*n)  # create array for points xvalue
    yval = np.linspace(0.0,1.0,2*n)  # create array for points yvalue
    x = x0
    for i in range(0,n):
        xval[2*i] = x  # first point is (x,f(x))
        x = LogisticMap(r,x)
        yval[2*i] = x
        xval[2*i+1] = x #second point is (f(x),f(x))
        yval[2*i+1] = x

    ax1.plot(xval,yval,'b')  # connect up all these points blue


parser = argparse.ArgumentParser(description='A simple code to iterate the logistic map')

parser.add_argument('-R',help="The parameter R (required)",type=float,required=True)
parser.add_argument('-x0',help="The initial seed (default 0.2)",type=float,default=0.2)
parser.add_argument('-iter',help="how many iterates (default 10)",type=int,default=10)

args = parser.parse_args()

ax1 = plt.subplot(121) # creates first axis
ax2 = plt.subplot(122)

fac=1.01
xmax = 1.00
xmin =-0.01
ymax = 1.01
ymin =-0.01
ax1.axis([xmin*fac,xmax*fac,ymin*fac,ymax*fac])
xvalues = np.arange(xmin, xmax, 0.01)   # to plot function

# parameters
r=args.R
niter=args.iter
X0 = args.x0

yvalues = LogisticMap(r,xvalues)           # function computed
ax1.plot(xvalues,yvalues, 'r')             # function plotted red
plot_cobweb(X0,r,niter)                    # cobweb plot
ax1.plot(xvalues,xvalues, 'g')               #y=x plotted green

x2 = []
y2= []

x2.append(0)
y2.append(X0)

for k in xrange(niter):
   X0=LogisticMap(r,X0)
   x2.append(k+1)
   y2.append(X0)

ax2.scatter(x2,y2)
plt.xlim(-0.1, float(niter)+.1)

plt.show()
