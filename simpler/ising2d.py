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

2D Ising Model Simulation with Simulated Annealing

This script simulates the behavior of a ferromagnetic material using the 2D Ising model, a statistical mechanics framework that models magnetic interactions between spins on a lattice. The simulation employs the Metropolis-Hastings algorithm for Monte Carlo updates, allowing spins to flip based on energy changes and thermal fluctuations. The temperature is gradually decreased from an initial high value (TEMP_START) to a final low value (TEMP_END) using a geometric cooling schedule, mimicking the process of simulated annealing. This approach helps the system converge to a stable magnetic configuration, illustrating how thermal energy influences the formation of magnetic domains.

Usage:

	# python3 ising2d.py

'''

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

# function to create a checkerboard pattern array with spins 1 and -1
def checkerboard(shape):
    return np.where(np.indices(shape).sum(axis=0) % 2 == 0,-1,1)

#sum over 4 neighboors
@jit(nopython=True)
def sumN(S,i,j,NX,NY):
	return S[(i+1)%NX,j] + S[(i-1)%NX,j] + S[i,(j+1)%NY] + S[i,(j-1)%NY]

# metropolis algorithm for 1 temperature
@jit(nopython=True)
def metropolis(N_PER_TEMP,NX,NY,S,J,kB,TEMP_START):
	for _ in range(N_PER_TEMP):
		i = random.randint(0,NX-1)
		j = random.randint(0,NY-1)
		DE = 2*J*S[i,j]*sumN(S,i,j,NX,NY) # delta E
		if DE < 0:
			S[i,j] *= -1
		else:
			r = random.random()
			if r < math.exp(-DE/(kB*TEMP_START)):
				S[i,j] *= -1

# constants
NX = 150
NY = 150
J = 1
N_PER_TEMP = 20 * NX * NY
TEMP_START = 4
TEMP_END = 0.1
TEMP_FACTOR = 0.98
#kB = 1.380649e-23 # Boltzmann constant
kB = 1 # normalized Boltzmann constant

S_vec = []

def init():
	global t,S,M,T,m
	S = checkerboard((NX,NY))
	M = 0
	t = TEMP_START
	T = []
	m = []
	return S, M, t, T, m

N = int(math.log10(TEMP_END/TEMP_START)/math.log10(TEMP_FACTOR)) # number of iterations
init() # initial conditions

# compute
for n in range(N):
	global showf,t
	
	T.append(t)
	metropolis(N_PER_TEMP,NX,NY,S,J,kB,t)
	M = np.sum(S)
	m.append(M/(NX*NY))
	t *= TEMP_FACTOR
	
	S_vec.append(S.copy())
	
	print("[OK] Iteration ",n)

# show
for n in range(N):

	plt.clf()
	plt.imshow(S_vec[n], cmap="binary")
	plt.pause(0.001)
	
plt.show()
