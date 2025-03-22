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

2D Transport Equation Solver: CIR vs CTU Method Comparison

This script solves the 2D transport equation using two finite volume methods:

1. Courant-Isaacson-Rees (CIR)
2. Corner Transport Upwind (CTU)

The problem simulates the advection of a 2D Gaussian density distribution
in a square domain with periodic boundary conditions. The script compares
the performance of CIR and CTU methods by visualizing their solutions 
side-by-side.

Usage:

	# python3 circtu.py
	
'''

import matplotlib.pyplot as plt
import numpy as np

a = 1 # physical x velocity [m/s]
b = 1 # physical y velocity [m/s]
L = 5 # grid side distance [m]
T = 12 # physical time [s]
N = 200 # grid side
dx = L/N # grid spacing
# Courant numbers
dt1 = 0.0125
dt2 = 0.0249
Ca1 = a*dt1/dx
Cb1 = b*dt1/dx
Ca2 = a*dt2/dx
Cb2 = b*dt2/dx
assert (Ca1 + Cb1) <= 1 and Ca2 < 1 and Cb2 < 1,"Courant numbers outside stability limit"
p1 = np.zeros([N,N]) # density CIR
p2 = np.zeros([N,N]) # density CTU
p2_ = np.zeros([N,N])
# physical positions
x = np.linspace(0,L,N)
y = np.linspace(0,L,N)
X,Y = np.meshgrid(x,y)

#initial conditions
s = 0.5
for i in range(N):
	for j in range(N):
		p1[i][j] = np.exp(-(0.5*((i*dx - (1.5/L)*N*dx)**2)/(s*s) + 0.5*((j*dx - (1.5/L)*N*dx)**2)/(s*s)))
p2 = np.copy(p1)

t = 0 # initial time

# Set up the figure for side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.ion()  # Turn on interactive mode

# Initial plots
contour1 = ax1.contour(X, Y, p1, cmap="magma")
ax1.set_title("CIR")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.axis("scaled")

contour2 = ax2.contour(X, Y, p2, cmap="magma")
ax2.set_title("CTU")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.axis("scaled")

while t <= T:

	# CIR
	p1[1:-1,1:-1] = -Ca1*(p1[1:-1,1:-1] - p1[1:-1,0:-2]) - Cb1*(p1[1:-1,1:-1] - p1[0:-2,1:-1]) + p1[1:-1,1:-1]
	
	# # CTU
	p2_[1:-1,1:-1] = (1 - Ca2)*p2[1:-1,1:-1] + Ca2*p2[1:-1,0:-2]
	p2[1:-1,1:-1] = (1 - Cb2)*p2_[1:-1,1:-1] + Cb2*p2_[0:-2,1:-1]
	
	# # periodic BCs
	
	# edges
	p1[:,0] = p1[:,N-2]
	p1[:,N-1] = p1[:,1]
	p1[0,:] = p1[N-2,:]
	p1[N-1,:] = p1[1,:]
	# corners
	p1[0,0] = p1[N-2,N-2]
	p1[N-1,N-1] = p1[1,1]
	p1[0,N-1] = p1[N-2,1]
	p1[N-1,0] = p1[1,N-2]
	
	# edges
	p2[:,0] = p2[:,N-2]
	p2[:,N-1] = p2[:,1]
	p2[0,:] = p2[N-2,:]
	p2[N-1,:] = p2[1,:]
	p2_[:,0] = p2_[:,N-2]
	p2_[:,N-1] = p2_[:,1]
	p2_[0,:] = p2_[N-2,:]
	p2_[N-1,:] = p2_[1,:]
	# corners
	p2[0,0] = p2[N-2,N-2]
	p2[N-1,N-1] = p2[1,1]
	p2[0,N-1] = p2[N-2,1]
	p2[N-1,0] = p2[1,N-2]
	p2_[0,0] = p2_[N-2,N-2]
	p2_[N-1,N-1] = p2_[1,1]
	p2_[0,N-1] = p2_[N-2,1]
	p2_[N-1,0] = p2_[1,N-2]
    
	# Update plots
	for c in contour1.collections:
		c.remove()
	contour1 = ax1.contour(X, Y, p1, cmap="magma")
    
	for c in contour2.collections:
		c.remove()
	contour2 = ax2.contour(X, Y, p2, cmap="magma")
    
	plt.pause(0.01)
    
	t += dt2

plt.ioff()  # Turn off interactive mode
plt.show()

