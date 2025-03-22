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

Pendulum Motion Simulator and Visualizer

This script simulates the motion of a simple pendulum using numerical methods
and provides a dynamic visualization of its movement over time.

Usage:
    
    # python3 pendulum.py
    
'''


import matplotlib.pyplot as plt
import numpy as np
import math

def pendulum(p,q):
	global epsilon, m
	dpdt = -epsilon*math.sin(q)
	dqdt = p/m
	return (dpdt,dqdt)

# one step solvers

# symplectic
def leapfrog(p0,q0,h,odeSystem):
	q12 = q0 + 0.5*h*p0 # first drift
	dp, dq = odeSystem(p0, q12)
	p1 = p0 + h*dp # kick
	q1 = q12 + 0.5*h*p1 # second drift
	return (p1,q1)

def euler(p,q,h,odeSystem):
	dp,dq = odeSystem(p,q)
	p += h*dp
	q += h*dq
	return (p,q)

def midrungekutta(p0,q0,h,odeSystem):
	dp0,dq0 = odeSystem(p0,q0)
	dp,dq = odeSystem(p0 + 0.5*h*dp0,q0 + 0.5*h*dq0)
	p1 = p0 + h*dp
	q1 = q0 + h*dq
	return (p1,q1)

# iterator

def iterator(p0,q0,N,h,solver,sys):
	pn = p0
	qn = q0
	plist = [p0]
	qlist = [q0]
	for t in range(N):
		P,Q = solver(pn,qn,h,sys)
		pn = P
		qn = Q
		plist.append(P)
		qlist.append(Q)
	return (np.array(plist),np.array(qlist))

# main

h = 0.02 # step
g = 9.8 # gravitational acceleration
T = 1000 # number of iterations
n = np.arange(0,T+1)

# initial energy for the pendulum

m = 2.0 # mass
l = 0.3 # pendulum length
epsilon = m*g*l
q0 = m*g*l*math.cos(math.pi/8.0)
p0 = 0.0

# Solve the pendulum system using Euler's method
P, Q = iterator(p0, q0, T, h, euler, pendulum)

# calculate pendulum positions (x and y) based on angle Q

L = 1.0  # length of the pendulum rod
x_positions = L * np.sin(Q)  # x-coordinate of the pendulum bob
y_positions = -L * np.cos(Q)  # y-coordinate of the pendulum bob

# plot pendulum positions at each time step using a loop and plt.show()

plt.figure()
for i in range(len(x_positions)):
    plt.clf()  # Clear the figure for each step
    plt.plot([0, x_positions[i]], [0, y_positions[i]], 'k-', label="Rod")  # Pendulum rod
    plt.plot(x_positions[i], y_positions[i], 'ro', label="Bob")            # Pendulum bob
    plt.xlim(-L - 0.2, L + 0.2)                                            # Set x-axis limits
    plt.ylim(-L - 0.2, L + 0.2)                                            # Set y-axis limits
    plt.gca().set_aspect('equal', adjustable='box')                        # Equal aspect ratio for x and y axes
    plt.title("Pendulum Position")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    
    plt.pause(0.01)                                                        # Pause to simulate animation

plt.show()                                                                 # Show
