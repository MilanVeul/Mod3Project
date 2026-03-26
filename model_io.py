import numpy as np
from datetime import datetime

def read_data(file):
    """Reads a given csv file and extracts the indices, timestamps and values."""
    data = np.loadtxt(file, skiprows=1, delimiter=";", dtype=str)

    start_time = data[0,1]
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    signal = data[:,2].astype(int)

    avarage_out_invalid_measurements(signal)

    return signal, start_time

def set_zero_invalid_measurements(signal):
    """Sets invalid measurements to 0"""
    mask = (signal == 999999999)
    signal[mask] = 0

def avarage_out_invalid_measurements(signal):
    """Takes avarage of surrounding valid measurements for each false measurement."""
    for i, x in enumerate(signal):
        if x != 999999999: continue
        if i == 0:
            signal[i] = signal[i+1]
            continue

        next_valid_index = -1
        for j in range(i+1, len(signal)):
            if signal[j] != 999999999:
                next_valid_index = j
                break
        if next_valid_index == -1:
            signal[i] = signal[i + 1]
            continue

        spacing = next_valid_index - (i-1)
        signal[i] = (signal[next_valid_index]*(1/spacing) + signal[i-1]*(1-(1/spacing)))

def generate_cosine(N, dt):
    times = np.arange(N)*dt
    indices = np.arange(N)
    A = 100
    f = 1/50
    phi = 0
    values = A*np.cos(2*np.pi*f*times + phi)
    return indices, times, values

def save_model(model, filename="models/model.npz"):
    w, A, phi, mu = model
    np.savez(filename, omegas=w, amplitudes=A, arguments=phi, mean=mu)
    print(f"Model saved to {filename}")

def load_model(filename="models/model.npz"):
    data = np.load(filename)
    model = (data['omegas'], data['amplitudes'], data['arguments'], data['mean'])
    print(f"Model loaded from {filename}")
    return model