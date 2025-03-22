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

3D Brownian Motion Simulation and Visualization

This script simulates and visualizes 3D Brownian motion using matplotlib.

Usage:

	# python3 brownian.py

'''

import matplotlib.pyplot as plt
import numpy as np

# initial positions

x, y, z = 0.0, 0.0, 0.0

# set up parameters

num_steps = 500
mean = 0
std_dev = 1

# create a random number generator

rng = np.random.default_rng()

# set up the 3D plot once

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# set axis limits

ax.set_xlim(-35, 35)
ax.set_ylim(-35, 35)
ax.set_zlim(-35, 35)

# generate Brownian motion

for n in range(num_steps):
    x_old, y_old, z_old = x, y, z

    # generate new Δx, Δy, and Δz for each time step
    
    delta_x = rng.normal(loc=mean, scale=std_dev)
    delta_y = rng.normal(loc=mean, scale=std_dev)
    delta_z = rng.normal(loc=mean, scale=std_dev)
    
    # update position
    
    x += delta_x
    y += delta_y
    z += delta_z
    
    ax.plot([x_old, x], [y_old, y], [z_old, z], color='blue', linewidth=0.5)
    plt.pause(0.001)

plt.show()

