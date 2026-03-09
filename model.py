import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

dt = 10 # In minutes

def read_data(file):
    data = np.loadtxt(file, skiprows=1, delimiter=";", dtype=str)
    
    indices = data[:,0].astype(int)
    times = None
    # Parsing time is extremely slow, so only enable it if you are actually use it
    # parse = np.vectorize(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    # times = parse(data[:,1])

    tide_raw = data[:,2].astype(int)
    # Remove invalid measurements
    mask = (tide_raw != 999999999)
    tide = tide_raw[mask]
    indices = indices[mask]
    return indices, times, tide

def frequency_analysis(signal):
    mean = np.mean(signal)
    print(f'mean = {mean}')
    # Make sure mean is 0. I found this the easiest, as we dont have to deal 
    # with the zero frequency with an unusual spike
    signal = signal - mean

    N = len(signal)
    M = 2**int(np.ceil(np.log2(N)))

    # Perform FFT
    fhat = np.fft.fft(signal, M) 
    fhat = fhat[0:M//2+1] # only take positive frequencies
    # Extract relevant quantities
    amplitudes = (2/N * abs(fhat)) # multiply by two to make up for negative counterpart
    arguments = np.angle(fhat)
    freq_mins = np.arange(M//2 + 1)/(M*dt)
    return freq_mins, amplitudes, arguments, mean

def plot_frequencies(frequencies, amplitudes):
    period_hours = (1 / frequencies) / 60
    plt.plot(period_hours, amplitudes)
    plt.xlim(0, 100)
    plt.xlabel('Period (Hours)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

def build_prediction_model(frequencies, amplitudes, arguments, mean):
    num_waves = 1000

    # Neat little trick i found on stackoverflow (i dont entirely understand how it works):
    # Selects the K indices with the highest amplitudes
    # Runs in O(n) instead of O(nlog(n)), which would be complexity of sorting the array
    max_indices = np.argpartition(amplitudes, -num_waves)[-num_waves:]

    frequencies = frequencies[max_indices]
    amplitudes = amplitudes[max_indices]
    arguments = arguments[max_indices]
    omegas = 2*np.pi*frequencies

    return (omegas, amplitudes, arguments, mean)

def predict(model, t):
    w, A, phi, mu = model # omegas, amplitudes, arguments (angles)
    # Add back the mean we have subtracted from the start
    return mu + np.sum(A*np.cos(w*t + phi))

def main():
    indices, times, tide = read_data("walsoorden2004-2024.csv")
    freq_mins, amplitudes, arguments, mean = frequency_analysis(tide)
    # plot_frequencies(freq_mins, amplitudes)
    model = build_prediction_model(freq_mins, amplitudes, arguments, mean)
    t=0
    print(f't={t}: {predict(model, t)}')


# Run script
if __name__ == "__main__":
    main()