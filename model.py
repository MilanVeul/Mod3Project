import numpy as np
from datetime import datetime
from plot import *
import model_io as io
from scipy.signal import find_peaks, windows
from multiprocessing import Process

dt = 10 # In minutes

def frequency_analysis(signal):
    """Performs a frequency analysis of a given signal using FFT"""
    mean = np.mean(signal)
    # Make sure mean is 0. I found this the easiest, as we dont have to deal 
    # with the zero frequency with an unusual spike
    signal = signal - mean

    N = len(signal)
    # M = 2**int(np.ceil(np.log2(N)))
    M = 2**25
    # M = N

    # Apply window
    windowing = True
    if windowing:
        window = np.hanning(N)
        signal = signal * window 

    # Perform FFT
    fhat = np.fft.fft(signal, M) 
    fhat = fhat[0:M//2+1] # only take positive frequencies
    # Extract relevant quantities
    amplitudes = (2/N * abs(fhat)) # multiply by two to make up for negative counterpart
    arguments = np.angle(fhat)
    freq_mins = np.arange(M//2 + 1)/(M*dt)

    if windowing:  #Windowing halves the amplitudes
        amplitudes *= 2
    return freq_mins, amplitudes, arguments, mean

###############################

def get_max_amplitudes(amplitudes, N):
    """Simply takes the N biggest frequencies"""
    max_indices = np.argpartition(amplitudes, -N)[-N:]
    return max_indices

def get_peaks(amplitudes, N):
    # Find all local maxima
    peaks, _ = find_peaks(amplitudes, height=0) 
    # Get the top K from the ACTUAL peaks only
    top_peaks_indices = peaks[np.argsort(amplitudes[peaks])[-N:]]
    return top_peaks_indices


def build_model(frequencies, amplitudes, arguments, mean):
    """Creates a model of the given frequencies, amplitudes and arguments of the FFT, and mean of the signal"""
    
    indices = get_peaks(amplitudes, 100)

    selected_frequencies = frequencies[indices]
    selected_amplitudes = amplitudes[indices]
    selected_arguments = arguments[indices]
    omegas = 2*np.pi*selected_frequencies
    
    # plt.plot(frequencies, amplitudes)
    # plt.scatter(selected_frequencies, selected_amplitudes, color='red')
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # plt.show()

    return (omegas, selected_amplitudes, selected_arguments, mean)
    
###############################
def predict_single(model, t):
    """Predicts the signal for a given time t"""
    w, A, phi, mu = model # omegas, amplitudes, arguments (angles)
    return mu + np.sum(A*np.cos(w*t + phi))

def predict_array(model, t_values):
    w, A, phi, mu = model
    t_col = t_values[:, np.newaxis] 
    
    return mu + np.sum(A * np.cos(w * t_col + phi), axis=1)

def rmse(model, actual, k):
    """Computes the Root Mean Squared Error of the prediction model."""
    assert k <= len(actual)
    N = len(actual)
    indices = np.linspace(0, N-1, k).astype(int)

    prediction = predict_array(model, (indices*dt))
    actual_subsamples = actual[indices]
    errors = prediction - actual_subsamples
    rmse = np.sqrt(np.mean(errors**2))
    return rmse

def compute_spaced_accuracy(model, actual, dt, step=10, n=None):
    """Computes RMSE using predict_single."""
    if n is None:
        n = len(actual)
    rmse_values = []
    for k in range(step, n+1, step):
        errors = []
        for i in range(k):
            t = i * dt
            prediction = predict_single(model, t)
            errors.append(prediction - actual[i])
        errors = np.array(errors)
        rmse = np.sqrt(np.mean(errors**2))
        rmse_values.append(rmse)
    return rmse_values

def above_150cm(data, time):
    currently_above = False
    intervals = []
    t1, t2 = -1
    for i, x in enumerate(data):
        if currently_above:
            if x < 150:
                t2 = data[i-1]
                currently_above = False
                intervals += [time[i],time[i-1]]
        else:
            if x >= 150:
                t1 = x
                currently_above = True

        if currently_above:
            t2 = data[-1]
            intervals += [t1, t2]
    return intervals

def main():
    indices, times, tide = io.read_data("walsoorden2004-2024.csv")
    
    # indices, times, tide = io.generate_cosine(10000, dt)

    freq_mins, amplitudes, arguments, mean = frequency_analysis(tide)
    # start,stop = (0.0013415, 0.0013422)
    # plot_frequencies(freq_mins, amplitudes, start, stop)

    model = build_model(freq_mins, amplitudes, arguments, mean)

    prediction = predict_array(model, (np.arange(15000, 16000)*dt))
    plot_comparison(np.arange(15000, 16000), (mean + tide)[15000:16000], prediction)

    print(f"RMSE: {rmse(model, mean + tide, 10000):.3f}")


# Run script
if __name__ == "__main__":
    main()