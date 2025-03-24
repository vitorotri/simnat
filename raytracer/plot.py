import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('shaded.txt')
x = data[:,0]
y = data[:,1]
R = data[:,2]
G = data[:,3]
B = data[:,4]
image = np.zeros((int(max(x)) + 1, int(max(y)) + 1, 3), dtype = float)
for i in range(len(x)):
	image[int(x[i]), int(y[i])] = [R[i], G[i], B[i]]
	
plt.style.use('dark_background')
plt.imshow(image, interpolation = 'none', origin = 'lower')
plt.axis('off')
plt.show()
