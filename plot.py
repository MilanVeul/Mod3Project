import numpy as np
import matplotlib.pyplot as plt

def plot_frequencies(frequencies, amplitudes):
    period_hours = (1 / frequencies) / 60
    plt.plot(period_hours, amplitudes)
    plt.xlim(0, 100)
    plt.xlabel('Period (Hours)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

def plot_comparison(actual, predicted):
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