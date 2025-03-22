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

Kepler's Equation Solver and Orbit Visualizer

This script solves Kepler's equation using Newton's Method and visualizes
the resulting orbit in real-time. It demonstrates the motion of a body
in an elliptical orbit around a central mass.

Usage:

	# python3 newtonkepler.py
	
'''

import matplotlib.pyplot as plt
import numpy as np

def kepler(E, M, e):
    return E - e * np.sin(E) - M

def keplerprime(e, E):
    return 1.0 - e * np.cos(E)

def newton(f, fprime, x0, tol, e, M):
    xn = x0
    FF = tol + 1.0
    while abs(FF) > tol:
        FF = f(xn, M, e) / fprime(e, xn)
        xnn = xn - FF
        xn = xnn
    return xn

# Main script
t = 0.01  # time-step
TT = 1/t  # quantity of dots for a yearly period
e = 0.35  # eccentricity
a = 1     # semi-major axis

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b.')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Kepler's Equation: Orbital Visualization")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

xdata, ydata = [], []

# Live update function
def update_plot(i):
    M = 2 * np.pi * t * i
    E0 = M
    E = newton(kepler, keplerprime, E0, 0.001, e, M)
    x = a * np.cos(E) - a * e
    y = a * np.sqrt(1 - e * e) * np.sin(E)
    xdata.append(x)
    ydata.append(y)
    line.set_data(xdata, ydata)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Run the live plot
for i in range(int(TT)):
    update_plot(i)
    plt.pause(0.03)  # Pause to control the update rate

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the plot open after the animation is done

