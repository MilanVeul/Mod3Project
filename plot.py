import numpy as np
import matplotlib.pyplot as plt

def plot_frequencies(frequencies, amplitudes, start=-1, stop=-1):
    # period_hours = (1 / frequencies) / 60
    plt.plot(frequencies, amplitudes)
    if start != -1:
        plt.xlim(start, stop)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

def plot_comparison(times, actual, predicted):
    assert len(actual) == len(predicted)
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label='Actual Data')
    plt.plot(predicted, label='Model Prediction', linestyle='--')
    plt.title(f"Tidal Prediction vs Actual Measurements")
    plt.ylabel("Water Level")
    # plt.xlabel("Time")
    plt.legend()
    plt.grid(True)
    plt.show()