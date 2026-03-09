import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("walsoorden2004-2024.csv", skiprows=1, delimiter=";", dtype=str)

dt = 10 # In minutes
t = data[:,0].astype(int)
tide_raw = data[:,2].astype(int)

# Remove invalid measurements
mask = (tide_raw != 999999999)
tide = tide_raw[mask]
t = t[mask]

# Make sure mean is 0
tide = tide - np.mean(tide)
N = len(tide)
M = 2**int(np.ceil(np.log2(N)))

fhat = np.fft.fft(tide,M)*dt   
fhat = fhat[0:M//2+1] # only take positive frequencies

freq_mins = np.arange(M//2 + 1)/(M*dt)
period_hours = (1 / freq_mins) / 60


plt.plot(period_hours, abs(fhat))
plt.xlim(0, 100)
plt.xlabel('Period (Hours)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

