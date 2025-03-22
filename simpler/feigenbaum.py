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

Feigenbaum Diagram Generator

This script generates a Feigenbaum diagram (also known as a bifurcation diagram) 
for the logistic map. It visualizes the long-term behavior of the logistic 
equation x(n+1) = a * x(n) * (1 - x(n)) for different values of the parameter 'a'.

Usage:

	# python3 feigenbaum.py
	
'''

import math
import matplotlib.pyplot as plt
import numpy as np

def logistic(a,x,n):
	# correction so it does not compute at the extrema
	if x == 0.0:
		x = 0.005
	elif x == 1.0:
		x -= 0.005
	for i in range(n):
		xn = a*x*(1.0 - x)
		x = xn
	return x

# main

A = np.arange(0.0,4.0,0.005)
X = np.arange(0.0,1.0,0.005)

for j in range(len(X)):
	x0 = X[j]
	Xinf = np.array([logistic(i,x0,1000) for i in A])
	plt.plot(A,Xinf,"b.",markersize = 1)

plt.rcParams['text.usetex'] = True
plt.xlabel("a")
plt.ylabel("$X_{\infty}$")
#plt.savefig("feigenbaum.png")
plt.show()
