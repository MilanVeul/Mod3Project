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

def plot_comparison(actual, predicted, num_points=500):
    plt.figure(figsize=(12, 5))
    plt.plot(actual[:num_points], label='Actual Data')
    plt.plot(predicted[:num_points], label='Model Prediction', linestyle='--')
    plt.title(f"Tidal Prediction vs Actual Measurements (First {num_points} points)")
    plt.ylabel("Water Level")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True)
    plt.show()