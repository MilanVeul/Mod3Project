import numpy as np
from datetime import datetime
from plot import *
import model_io as io
from scipy.signal import find_peaks

dt = 10 # In minutes

def frequency_analysis(signal):
    """Performs a frequency analysis of a given signal using FFT"""
    mean = np.mean(signal)
    # Make sure mean is 0. I found this the easiest, as we dont have to deal 
    # with the zero frequency with an unusual spike
    signal = signal - mean

    N = len(signal)
    # M = 2**int(np.ceil(np.log2(N)))
    M = N

    # Apply window
    # window = np.hanning(N)
    # signal = signal * window 

    # Perform FFT
    fhat = np.fft.fft(signal, M) 
    fhat = fhat[0:M//2+1] # only take positive frequencies
    # Extract relevant quantities
    amplitudes = (2/N * abs(fhat)) # multiply by two to make up for negative counterpart
    arguments = np.angle(fhat)
    freq_mins = np.arange(M//2 + 1)/(M*dt)
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
    
    indices = get_peaks(amplitudes, 1000)

    frequencies = frequencies[indices]
    amplitudes = amplitudes[indices]
    arguments = arguments[indices]
    omegas = 2*np.pi*frequencies

    return (omegas, amplitudes, arguments, mean)
    
###############################
def predict_single(model, t):
    """Predicts the signal for a given time t"""
    w, A, phi, mu = model # omegas, amplitudes, arguments (angles)
    return mu + np.sum(A*np.cos(w*t + phi))

def predict_array(model, t_values):
    w, A, phi, mu = model
    t_col = t_values[:, np.newaxis] 
    
    return mu + np.sum(A * np.cos(w * t_col + phi), axis=1)

def compute_accuracy(model, actual, k):
    """Computes the Root Mean Squared Error of the prediction model."""
    assert k <= len(actual)
    N = len(actual)
    indices = np.linspace(0, N-1, k).astype(int)

    prediction = predict_array(model, indices*dt)
    actual_subsamples = actual[indices]
    errors = prediction - actual_subsamples
    rmse = np.sqrt(np.mean(errors**2))
    return rmse

def main():
    indices, times, tide = io.read_data("walsoorden2004-2024.csv")
    freq_mins, amplitudes, arguments, mean = frequency_analysis(tide)
    # plot_frequencies(freq_mins, amplitudes)
    model = build_model(freq_mins, amplitudes, arguments, mean)

    print(f"RMSE: {compute_accuracy(model, mean + tide, 100000):.3f}")


# Run script
if __name__ == "__main__":
    main()