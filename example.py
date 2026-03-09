import numpy as np
Tepoch = 1
Ts = 0.01
t = np.arange(0,Tepoch+Ts/2,Ts)
f = np.cos(2*np.pi*5*t)+np.cos(2*np.pi*10*t)/2
N = len(f)
M = 2**5
w = np.arange((M//2)+1)*2*np.pi/M/Ts
fhat = np.fft.fft(f,M)*Ts
fhat = fhat[0:(M//2+1)]

import matplotlib.pyplot as plt
plt.plot(w/(2*np.pi), abs(fhat))
# plt.rc('text', usetex=True) # optional
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{fourier}'] # optional
# plt.rc('font', family='serif', size=18) # optional
plt.xlabel('frequency $\omega/(2\pi)$ [t]$^{-1}$')
plt.ylabel('unit: $[f][t]$')
plt.grid()
# plt.savefig('code1p1.pdf')
plt.show()