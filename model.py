import numpy as np
from datetime import datetime
from plot import *
import model_io as io

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

    # Perform FFT
    fhat = np.fft.fft(signal, M) 
    fhat = fhat[0:M//2+1] # only take positive frequencies
    # Extract relevant quantities
    amplitudes = (2/N * abs(fhat)) # multiply by two to make up for negative counterpart
    arguments = np.angle(fhat)
    freq_mins = np.arange(M//2 + 1)/(M*dt)
    return freq_mins, amplitudes, arguments, mean

def build_model(frequencies, amplitudes, arguments, mean):
    """Creates a model of the given frequencies, amplitudes and arguments of the FFT, and mean of the signal"""
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

def predict(model, k, dt):
    """Predicts the first k values using the given model"""
    t = np.arange(k)*dt
    w, A, phi, mu = model # omegas, amplitudes, arguments (angles)
    return mu + np.sum(A[:, np.newaxis] * np.cos(w[:, np.newaxis] * t + phi[:, np.newaxis]), axis=0)
    
def predict_single(model, t):
    """Predicts the signal for a given time t"""
    w, A, phi, mu = model # omegas, amplitudes, arguments (angles)
    return mu + np.sum(A*np.cos(w*t + phi))

def compute_accuracy(model, original, dt, k):
    """Computes the Mean Squared Error of the prediction model."""
    assert k <= len(original)
    prediction = predict(model, k, dt)
    original = original[:k]
    errors = prediction - original
    rmse = np.sqrt(np.mean(errors**2))
    return rmse

def main():
    indices, times, tide = io.read_data("walsoorden2004-2024.csv")
    # freq_mins, amplitudes, arguments, mean = frequency_analysis(tide)
    # plot_frequencies(freq_mins, amplitudes)
    # model = build_model(freq_mins, amplitudes, arguments, mean)
    model = io.load_model()
    mean = model[3]
    print(f"RMSE = {compute_accuracy(model, mean + tide, dt, 10000)}")

    # io.save_model(model)

    # prediction_size = 2000
    # prediction = predict(model, prediction_size, dt)
    # plot_comparison(mean + tide, prediction, prediction_size)


# Run script
if __name__ == "__main__":
    main()