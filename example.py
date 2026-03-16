import numpy as np
Tepoch = 1
Ts = 0.01
t = np.arange(0,Tepoch+Ts/2,Ts)
f = np.cos(5*(2*np.pi)*t)

N = len(f)
M = 2**10
# M = N
w = np.arange((M//2)+1)*(2*np.pi)/M/Ts

window = np.hanning(N)
f = f * window

fhat = np.fft.fft(f,M)*Ts
fhat = fhat[0:(M//2+1)]

import matplotlib.pyplot as plt
plt.plot(w/(2*np.pi), 2*2*abs(fhat))
# plt.rc('text', usetex=True) # optional
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{fourier}'] # optional
# plt.rc('font', family='serif', size=18) # optional
plt.xlabel('frequency $\omega/(2\pi)$ [t]$^{-1}$')
plt.ylabel('unit: $[f][t]$')
plt.xlim(0, 10)
plt.grid()
# plt.savefig('code1p1.pdf')
plt.show()