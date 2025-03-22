# BSD 0-Clause License

# Copyright (c) 2025 Vito Romanelli Tricanico

# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

'''

Lotka-Volterra Predator-Prey Model Simulator

This script simulates the Lotka-Volterra predator-prey model, also known as the 
predator-prey equations. It models the dynamics between two species in an 
ecosystem: predators (foxes) and prey (mice).

The script can use both the Forward Euler method and the 4th Order Runge-Kutta 
method to solve the differential equations numerically. It then plots the 
results in two formats:

1. Phase space diagram (Foxes vs Mice)
2. Population time dependence (Foxes and Mice vs Time)

Parameters (can be adjusted in the script):
- h: Step size for numerical integration (default: 0.01)
- km, kmf, kfm, kf: Rate constants for the Lotka-Volterra equations
- m0, f0: Initial populations of mice and foxes
- T: Total simulation time/generations

Usage:
    
    # python3 lotkavolterra.py

'''

import matplotlib.pyplot as plt
import numpy as np

# one-step Euler solver
def forwardEuler(dfdt,h,fn,a):
	fnn = fn + h*dfdt(fn,a)
	return fnn

def dfoxdt(f,m):
	global kf, kfm
	return -kf*f + kfm*m*f
	
def dmousedt(m,f):
	global km, kmf
	return km*m - kmf*m*f

def lotkavolterra(kf,km,kfm,kmf,f0,m0,N,h,solver):
	fn = f0
	mn = m0
	flist = [f0]
	mlist = [m0]
	for t in range(N):
		FOX = solver(dfoxdt,h,fn,mn)
		MOUSE = solver(dmousedt,h,mn,fn)
		fn = FOX
		mn = MOUSE
		flist.append(FOX)
		mlist.append(MOUSE)
	return (np.array(flist), np.array(mlist))

# one-step 4th Order Runge-Kutta solver
def rungekutta4(dfdt,h,fn,a):
	k1 = h*dfdt(fn,a);
	k2 = h*dfdt(fn + 0.5*k1,a)
	k3 = h*dfdt(fn + 0.5*k2,a)
	k4 = h*dfdt(fn + k3,a)
	fnn = fn + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
	return fnn

# main

h = 0.01 # step

km = 2.0
kmf = 0.02
kfm = 0.01
kf = 1.06

m0 = 100
f0 = 15

T = 1000 # time/generations

n = np.arange(0,T+1)

F,M = lotkavolterra(kf,km,kfm,kmf,f0,m0,T,h,rungekutta4)

plt.figure(1)

# phase space
plt.plot(F,M,"-b")
plt.xlabel("Foxes")
plt.ylabel("Mice")
#plt.savefig("phasediagram.png")

plt.figure(2)

# populations x time
plt.plot(n,F,"-g")
plt.plot(n,M,"-r")
plt.legend(["Foxes","Mice"])
plt.xlabel("Time")
plt.ylabel("Populations")
#plt.savefig("timedependence.png")

plt.show()
