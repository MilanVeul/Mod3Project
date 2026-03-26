import numpy as np
import datetime
from plot import *
import model_io as io
from scipy.signal import find_peaks

dt = 10 # In minutes

class TideModel:
    signal = 0
    validation_signal = 0
    signal_mean = 0

    trainings_interval = []
    validation_interval = []

    start_time: datetime.datetime = None
    windowing_enabled = False

    frequencies = None
    amplitudes = None
    arguments = None

    def __init__(self, total_signal, start_time: datetime.datetime, training_ratio, windowing=False):
        split_index = int(len(total_signal) * training_ratio)
        self.signal = total_signal[:split_index]
        self.validation_signal = total_signal[split_index:]

        self.validation_interval = [split_index, len(total_signal)]
        self.trainings_interval = [0, split_index]

        self.start_time = start_time

        self.windowing_enabled = windowing
        self.frequency_analysis()
        self.build_model()

    def get_signal(self):
        return self.signal + self.signal_mean
    def get_total_signal(self):
        return np.concatenate((self.get_signal(), self.validation_signal))

    def frequency_analysis(self):
        """Performs a frequency analysis of a given signal using FFT"""
        self.signal_mean = np.mean(self.signal)
        # Make sure mean is 0. I found this the easiest, as we dont have to deal 
        # with the zero frequency with an unusual spike
        self.signal = self.signal - self.signal_mean

        N = len(self.signal)
        # M = 2**int(np.ceil(np.log2(N)))
        M = 2**25

        # Apply window
        processed_signal = self.signal
        if self.windowing_enabled:
            window = np.hanning(N)
            processed_signal = self.signal * window 

        # Perform FFT
        fhat = np.fft.fft(processed_signal, M) 
        fhat = fhat[0:M//2+1] # only take positive frequencies
        # Extract relevant quantities
        self.amplitudes = (2/N * abs(fhat)) # multiply by two to make up for negative counterpart
        self.arguments = np.angle(fhat)
        self.frequencies = np.arange(M//2 + 1)/(M*dt)

        if self.windowing_enabled:  #Windowing halves the amplitudes
            self.amplitudes *= 2

    def build_model(self):
        indices = get_peaks(self.amplitudes, 30)
        self.frequencies = self.frequencies[indices]
        self.amplitudes = self.amplitudes[indices]
        self.arguments = self.arguments[indices]
        self.omegas = 2*np.pi*self.frequencies
        
        # plt.plot(frequencies, amplitudes)
        # plt.scatter(selected_frequencies, selected_amplitudes, color='red')
        # plt.xlabel('Frequency')
        # plt.ylabel('Amplitude')
        # plt.grid()
        # plt.show()

    def predict_array(self, t_values):
        t_col = t_values[:, np.newaxis] 
        return self.signal_mean + np.sum(self.amplitudes * np.cos(self.omegas * t_col + self.arguments), axis=1)
    def predict_single(self, t):
        """Predicts the tide for a given time t"""
        return self.signal_mean + np.sum(self.amplitudes*np.cos(self.omegas*t + self.arguments))

####################

def above_150(data):
    currently_above = False
    intervals = []
    i1 = -1
    i2 = -1
    for i, x in enumerate(data):
        if currently_above:
            if x < 150:
                i2 = i-1
                currently_above = False
                intervals.append([i1,i2])
        else:
            if x >= 150:
                i1 = i
                currently_above = True

    if currently_above:
        i2 = len(data)-1
        intervals.append([i1, i2])
    return intervals

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

###############################

def rmse(model: TideModel, k):
    """Computes the Root Mean Squared Error of the prediction model."""
    true_signal = model.get_total_signal()
    
    assert k <= model.validation_interval[1] - model.validation_interval[0]
    
    pred_indices = np.linspace(model.validation_interval[0], model.validation_interval[1]-1, k).astype(int)
    pred_times = pred_indices * dt

    prediction = model.predict_array(pred_times)
    true_signal_subsamples = true_signal[pred_indices]
    errors = prediction - true_signal_subsamples
    rmse = np.sqrt(np.mean(errors**2))
    return rmse

##################

def index_to_time(start_time, index):
    return start_time + datetime.timedelta(minutes=index*dt)
def time_to_index(start_time, index):
    return start_time - datetime.timedelta(minutes=index*dt)


def main():
    raw_signal, start_time = io.read_data("walsoorden2004-2024.csv")

    model = TideModel(raw_signal, start_time, 0.9, windowing=True)

    # times = np.arange(model.validation_interval[0], model.validation_interval[1]-1)
    prediction = model.predict_array(np.arange(0, 500)*dt)

    print(above_150(prediction))
    print(above_150(model.validation_signal))

    # plot_comparison(model.get_total_signal()[0:500], prediction)

    print(f"RMSE: {rmse(model, 10000):.3f}")

# Run script
if __name__ == "__main__":
    main()