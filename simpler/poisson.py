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

Poisson Equation Solver using Successive Over-Relaxation (SOR) Method

This script solves the 2D Poisson equation using the Successive Over-Relaxation 
(SOR) method, which is an improvement over the Jacobi method. It then visualizes 
the field using contour plots, considering a plate with some voltage in the 
middle of the domain.

Usage:
    
    # python3 poisson.py
    
'''

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage

# N = 100 # Jacobi
# omega = 1 # Jacobi
N = 100 # SOR
omega = 2.0/(1 + (math.pi/N)) # SOR
L = 1.0

x = np.linspace(0,L,N)
y = np.linspace(0,L,N)
X, Y = np.meshgrid(x,y)

phi = np.zeros([N,N])
phi[math.ceil(0.5*N),math.ceil(0.25*N):math.ceil(0.75*N)] = 1000 # voltage
R = np.zeros([N,N])
R[1:-1,1:-1] = omega/4
R[math.ceil(0.5*N),math.ceil(0.25*N):math.ceil(0.75*N)] = 0
W = np.array([[0,1,0],[1,-4,1],[0,1,0]]) # stencil weights
aux = np.zeros([N,N])

P = np.ones_like(phi,dtype=bool)
P[::2,::2] = False
P[1::2,1::2] = False

# plt.pcolormesh(R)
# plt.show()

# p = 1
# N_iter = 2*p*N**4 # Jacobi
p = 3
N_iter = math.floor((1/3)*p*N) # SOR

for i in range(N_iter):
	ndimage.convolve(phi,W,output = aux,mode = "constant",cval = 0)
	M = np.multiply(R,aux)
	phi[~P] = phi[~P] + M[~P]
	ndimage.convolve(phi,W,output = aux,mode = "constant",cval = 0)
	M = np.multiply(R,aux)
	phi[P] = phi[P] + M[P]
	
CS = plt.contour(X,Y,phi,[5,50,100,250,500,750,950])
# CS = plt.contour(X,Y,phi)
plt.clabel(CS,inline=True, fontsize=7)
#plt.savefig("poisson.png")
plt.show()
