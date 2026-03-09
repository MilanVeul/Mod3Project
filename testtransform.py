import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("walsoorden2004-2024.csv", skiprows=1, delimiter=";", dtype=str)
# print(data)

dt = 1

t = data[:,0].astype(int)
f = data[:,2].astype(int)
print(f)
N = len(f)
M = 2**11
# w = np.arange((M//2)+1)*2*np.pi/M/dt
fhat = np.fft.fft(f,N)*dt
# print(fhat)
# exit(0)
fhat = fhat[0:N]
print(fhat)

plt.plot(t/(2*np.pi), abs(fhat))
plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{fourier}'] # optional

plt.xlabel('frequency')
plt.ylabel('unit')
plt.grid()
plt.show()