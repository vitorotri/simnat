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

Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT) Implementation

This script demonstrates the implementation of two Fourier transform algorithms
for a given sum of sinusoid signals:

1. A simple Discrete Fourier Transform (DFT)
2. The Cooley-Tukey Fast Fourier Transform (FFT)

The script also compares the implemented FFT with NumPy's built-in FFT function.

Usage:

	python3 myFFT.py

'''

import numpy as np
import matplotlib.pyplot as plt

# simple DFT algorithm
# x -> input array of size 2**i
def DFT(x):
	N = len(x)
	p, q = np.meshgrid(np.arange(N), np.arange(N))
	W_n = np.exp(-2j*np.pi*p*q/N)
	X = np.dot(W_n, x)
	return X

# Cooley-Tukey FFT algorithm
# x -> input array of size 2**i
def FFT(x):
	N = len(x)
	if N <= 1:
		return x
	else:
		X_even = FFT(x[::2])
		X_odd = FFT(x[1::2])
		diag_w = np.exp(2j*np.pi*np.arange(N//2)/N)
		X = np.zeros(N, dtype=complex)
		X[:N//2] = X_even + diag_w*X_odd
		X[N//2:] = X_even - diag_w*X_odd
		return X
	

i = 10
N = 2**i # if i = 10, then N = 1024
L, Delta = np.linspace(0, 1, N, retstep=True) # normalized linear space

fc = 1/(2*Delta) # Nyquist frequency [Hz]
fn = np.arange(0, fc, fc/N*2)

x = np.sin(2*np.pi*10*L) # sinusoid input at 10 Hz
x += np.sin(2*np.pi*200*L) # sinusoid input at 30 Hz
x += np.sin(2*np.pi*450*L) # sinusoid input at 50 Hz

#y = DFT(x)[:N//2]
y = FFT(x)[:N//2]
y_np = np.fft.fft(x)[:N//2]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(L, x, 'r')
axs[0].set_xlabel('x [n]')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Resulting Signal')
axs[1].plot(fn, abs(y), 'b', label='Implemented')
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Amplitude')
axs[1].plot(fn, abs(y_np), 'g.', label='Numpy')
axs[1].set_title('Power Spectrum')
plt.legend()
plt.show()

