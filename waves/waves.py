# Copyright (c) 2025 Vito Romanelli Tricanico
#
# See the LICENSE file for rights and limitations.

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
import matplotlib.cm as cm
import math
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

if __name__ == "__main__":
	Lx = 1.0 # domain size in x
	Ly = 1.0 # domain size in y
	dx = 0.025 # mesh size

	# select pulse type

	input1 = input("\nEnter initial conditions:\n\n(1) Single Centered Gaussian\n(2) Two Gaussians\n(3) Local Delta Pulse\n" + "(4) Top Hat Pulse\n(5) Ring Pulse\n(6) Asymetric Gaussian\n\n")
	input1 = int(input1)
	if input1 < 1 or input1 > 6:
		print("\n[!] Unknown input: selecting single centered gaussian\n\n")
		input1 = 1
	
	# select attenuation
	
	input2 =  input("\nSelect an option:\n\n(1) Regular\n(2) Attenuated\n\n")
	input2 = int(input2) - 1
	if input2 < 0 or input2 > 1:
		print("\n[!] Unknown input: selecting (1) Regular option\n\n")
		input2 = 0
		
	# Types reflection coefficient
	
	input_R =  input("\nProvide the reflection coefficient (ranges from -1 to 1)\n\n")
	input_R = float(input_R)
	if input_R < -1.0 or input_R > 1.0:
		print("\n[!] Incorrect input: setting R = 0.0\n\n")
		input_R = 0.0

	# Include obstacles or not
	
	input3 =  input("\nInclude some obstacles?\n\n(1) Yes\n(2) No\n\n")
	input3 = int(input3) - 1
	if input3 < 0 or input3 > 1:
		print("\n[!] Incorrect input: selecting (2) No\n\n")
		input3 = 1

	c = 1.0 # normalized speed of sound
	CFL = 1.0/math.sqrt(2) # Courant-Friedrichs-Lewy condition
	dt = CFL*dx/c # time-step (threshold)

	X = int(Lx/dx) # domain size index in x
	Y = int(Ly/dx) # domain size index in y

	x = np.arange(0, Lx, dx) # grid array for x
	y = np.arange(0, Ly, dx) # grid array for y
	XX, YY = np.meshgrid(x, y)
	A = 1.0 # initial amplitude of signal
	p2 = np.zeros((X,Y)) # pressure at p[n+1]
	p1 = np.copy(p2) # pressure at p[n]
	p0 = np.copy(p2) # pressure at p[n-1]
	
	# mask for obstacles
	obstacle_mask = np.ones((X, Y))
	obstacle_mask[int(0.25*X):int(0.55*X), int(0.25*Y):int(0.55*Y)] = 0
	obstacle_mask[int(0.60*X):int(0.85*X), int(0.60*Y):int(0.85*Y)] = 0
	obstacle_mask[int(0.75*X):int(0.85*X), int(0.25*Y):int(0.55*Y)] = 0
		
	# use a single expression that selects computed_mask if input3==0, or a full ones mask if not.
	obstacle_mask = obstacle_mask * int(input3 == 0) + np.ones((X, Y)) * int(input3 != 0)

	### source and time-step

	src_x = 0.5*Lx # source x position
	src_y = 0.5*Ly # source y position
	rec_x = 0.95*Lx # receiver x position (for impulse response)
	rec_y = 0.95*Ly # receiver y position (for impulse response)

	src = [int(src_x/dx),int(src_y/dx)] # source array
	rec = [int(rec_x/dx),int(rec_y/dx)] # receiver array

	t = 0.0 # initial time
	t_max = 5.0 # end time
	N = int((t_max - t)/dt) # number of time samples

	pt = np.zeros(N) # array for impulse response at receiver position

	##### initial conditions #####
	# (1) centered gaussian pulse (big profile)
	if input1 == 1:
		p0 = np.array([[A*math.exp(-((i - src_x)**2 + (j - src_y)**2)/(50*dx*dx)) for i in x] for j in y])
		p1 = np.copy(p0)

	# (2) 2 gaussian pulses (small profile)
	elif input1 == 2:
		p0 = np.array([[A*math.exp(-((i - 0.25*Lx)**2 + (j - 0.25*Ly)**2)/(5*dx*dx)) for i in x] for j in y])
		p0 += np.array([[A*math.exp(-((i - 0.75*Lx)**2 + (j - 0.75*Ly)**2)/(5*dx*dx)) for i in x] for j in y]) # for interference
		p1 = np.copy(p0)

	# (3) local delta pulse
	elif input1 == 3:
		p0[src[0],src[1]] = A
		#p0[int((Lx / 4.0) / dx), int((Lx / 4.0) / dx)] = A  # for interference
		p1 = np.copy(p0)

	# (4) top hat pulse
	elif input1 == 4:
		radius = 0.2
		p0 = np.array([[0.1*A if math.sqrt(((i - src_x)**2 + (j - src_y)**2)) <= radius else 0 for i in x] for j in y])
		p1 = np.copy(p0)

	# (5) ring pulse
	elif input1 == 5:
		inner_radius = 0.27
		outer_radius = 0.3
		p0 = np.array([[0.1*A if inner_radius <= math.sqrt((i - src_x)**2 + (j - src_y)**2) <= outer_radius else 0 for i in x] for j in y])
		p1 = np.copy(p0)
	
	# (6) Asymetric Gaussian (tsunami) pulse
	elif input1 == 6:
		p0 = np.array([[A*math.exp(-(((i - 0.2*Lx)**2)/200 + (j - 0.1*Ly)**2)/(2*dx*dx)) for i in x] for j in y])
		p1 = np.copy(p0)

	##### plots power spectrum of initial signal #####
	fig0 = plt.figure()

	FFN = np.fft.fft2(p0)
	FFN = np.copy(FFN[:int(len(FFN)/2)])
	mag = np.abs(FFN)
	freq = np.arange(len(FFN))/(2*len(FFN))
	f = interp1d(freq,mag[:,0],kind="cubic")
	freq_interp = np.linspace(freq[0],freq[-1],200)
	mag_interp = f(freq_interp)
	
	# Set the position of the first figure
	
	manager1 = plt.get_current_fig_manager()
	manager1.window.wm_geometry("+300+250")  # "+x_pos+y_pos"
	
	# plt.plot(freq,mag[:,0],"b")
	plt.plot(freq_interp,mag_interp,"g")
	plt.xlabel("Normalized frequency [Hz]")
	plt.ylabel("Power Spectral Density")

	#### Create second figure and axes

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	mcmap = "magma"
	surf = ax.plot_surface(XX, YY, p2, cmap=mcmap, zorder=1)
	ax.set_zlim(-1, 1)		
	
	# Add labels to the axes
	ax.set_xlabel('X (m)')
	ax.set_ylabel('Y (m)')
	ax.set_zlabel('Amplitude')
	
	# Set the position of the second figure
	
	manager2 = plt.get_current_fig_manager()
	manager2.window.wm_geometry("+1000+250")  # "+x_pos+y_pos"

	# attenuation factor (should be positive)
	alpha = 0.1

	
	
	#### time loop
	
	n = 0
	while n < N:

		# update equation
		p2[1:-1,1:-1] = 0.5*(p1[2:,1:-1] + p1[0:-2,1:-1] + p1[1:-1,2:] + p1[1:-1,0:-2]) - p0[1:-1,1:-1]
		# apply mask after update
		p2[obstacle_mask == 0] = 0
		
		### Boundary Conditions
			
		# absorbing
		R = input_R # reflection coefficient
		p2[0,:] = (1 + R)*p1[1,:] - R*p0[0,:]
		p2[:,0] = (1 + R)*p1[:,1] - R*p0[:,0]
		p2[X-1,:] = (1 + R)*p1[X-2,:] - R*p0[X-1,:]
		p2[:,Y-1] = (1 + R)*p1[:,Y-2] - R*p0[:,Y-1]
		
		# free ends (same as above with R = -1)
		#p2[0,:] = p2[1,:]
		#p2[:,0] = p2[:,1]
		#p2[X-1,:] = p2[X-2,:]
		#p2[:,Y-1] = p2[:,Y-2]
		

		# apply attenuation
		p2 *= (1 - input2) + input2*math.exp(-alpha*dt*n)

		### Impulse Response
		pt[n] = np.copy(p2[rec[0]][rec[1]])
		# pt[n] = p2[rec[0], rec[1]] - p0[rec[0], rec[1]] # removes background signal if p0/p1 = 0 (FDTD)

		### update the plot data
		
		surf.remove()
		surf = ax.plot_surface(XX, YY, p2, cmap=mcmap, zorder=1)
		
		plt.pause(0.01)

		### update arrays + time-step
		p0 = np.copy(p1)
		p1 = np.copy(p2)
		n += 1

	#### plot impulse response
	
	fig2 = plt.figure()
	plt.plot(pt,"b")
	plt.xlabel("Samples [n]")
	plt.ylabel("Pressure")
	plt.title("Impulse Response at [rec_x, rec_y]")
	
	
	
	'''
	# Update function for FuncAnimation (comment section above if using this)
	n = 0
	def update(frame):
		global p0, p1, p2, n
    	
		# update equation
		p2[1:-1,1:-1] = 0.5*(p1[2:,1:-1] + p1[0:-2,1:-1] + p1[1:-1,2:] + p1[1:-1,0:-2]) - p0[1:-1,1:-1]
		# apply mask after update
		p2[obstacle_mask == 0] = 0
    	
		### Boundary Conditions
			
		# absorbing
		R = input_R # reflection coefficient
		p2[0,:] = (1 + R)*p1[1,:] - R*p0[0,:]
		p2[:,0] = (1 + R)*p1[:,1] - R*p0[:,0]
		p2[X-1,:] = (1 + R)*p1[X-2,:] - R*p0[X-1,:]
		p2[:,Y-1] = (1 + R)*p1[:,Y-2] - R*p0[:,Y-1]
		
		# apply attenuation
		p2 *= (1 - input2) + input2*math.exp(-alpha*dt*n)
	
		# Update plot data
		ax.clear()
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Amplitude')
		ax.set_zlim(-A, A)
		surf = ax.plot_surface(XX, YY, p2, cmap="magma")
    	
		# Update arrays for next time step
		p0 = np.copy(p1)
		p1 = np.copy(p2)
		n += 1
	
	# Create animation using FuncAnimation
	ani = FuncAnimation(fig, update, frames=N)
	
	# Save the animation as a video file using FFMpegWriter
	writer = FFMpegWriter(fps=15)
	ani.save("anim.mp4", writer=writer)
	'''
	
	
	plt.show()
