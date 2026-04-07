import numpy as np
from datetime import datetime
from plot import *
import model_io as io
from scipy.signal import find_peaks

dt = 10 # In minutes
NUM_WAVES = 20

class TideModel:
    signal = 0
    validation_signal = 0
    mean = 0

    trainings_interval = []
    validation_interval = []

    start_time: datetime = None
    windowing_enabled = True

    frequencies = None
    amplitudes = None
    arguments = None

    def __init__(self, total_signal, start_time: datetime, training_ratio, windowing=False):
        split_index = int(len(total_signal) * training_ratio)
        self.signal = total_signal[:split_index]
        self.validation_signal = total_signal[split_index:]

        self.validation_interval = [split_index, len(total_signal)]
        self.trainings_interval = [0, split_index]

        self.start_time = start_time

        self.windowing_enabled = windowing
        self.frequency_analysis()
        self.build_model()

    def get_total_signal(self):
        return np.concatenate((self.mean + self.signal, self.validation_signal))

    def frequency_analysis(self):
        """Performs a frequency analysis of a given signal using FFT"""

        self.mean = 0
        self.mean = np.mean(self.signal)
        self.signal = self.signal - self.mean
        
        N = len(self.signal)
        # M = N
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
        self.amplitudes[0] = 0 # Since we subtracted the mean
        self.arguments = np.angle(fhat)
        self.frequencies = np.arange(M//2 + 1)/(M*dt)

        if self.windowing_enabled:  # Windowing halves the amplitudes
            self.amplitudes *= 2    

    def build_model(self):
        indices = get_peaks(self.amplitudes, NUM_WAVES)
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
        self.print_max_freqs()


    def print_max_freqs(self):
        # Get indices of the 10 largest amplitudes
        top_10_indices = np.argsort(self.amplitudes)[-10:][::-1]

        print(f"{'Rank':<5} | {'Freq (1/day)':<12} | {'Period (hours)':<15} | {'Amplitude (cm)':<15}")
        print("-" * 55)

        for i, idx in enumerate(top_10_indices, 1):
            freq = self.frequencies[idx] * 60 * 24
            amp = self.amplitudes[idx]
            
            # Calculate period in hours (1 / (freq * 60))
            period_hours = 1 / (freq / 24) if freq != 0 else float('inf')
            
            print(f"{i:<5} | {freq:<12.2f} | {period_hours:<15.2f} | {amp:<15.2f}")
    
    def get_signal(self, index_times):
        signal = self.signal + self.mean
        indices = index_times/dt
        s = []
        for i in indices:
            t = (i - np.floor(i))
            val = signal[int(np.floor(i))]*(1-t) + signal[int(np.ceil(i))]*t
            s.append(val)
        return np.array(s)

    def predict_array(self, t_values):
        t_col = t_values[:, np.newaxis] 
        return self.mean + np.sum(self.amplitudes * np.cos(self.omegas * t_col + self.arguments), axis=1)
    def predict_single(self, t):
        """Predicts the tide for a given time t"""
        return self.mean + np.sum(self.amplitudes*np.cos(self.omegas*t + self.arguments))

####################
def compare_windows(model: TideModel):
    training_length = len(model.signal)
    times = np.arange(training_length, training_length + len(model.validation_signal)-1, step=1)
    prediction = model.predict_array(times)
    pred_windows = accessible_windows(np.column_stack((times, prediction)))
    actual_windows = accessible_windows(np.column_stack((times, model.get_signal(times))))
    pred_durations = pred_windows[:,1] - pred_windows[:,0]
    actual_durations = actual_windows[:,1] - actual_windows[:,0]

    min_len = min(len(pred_durations), len(actual_durations))

    return rmse(
        pred_durations[:min_len],
        actual_durations[:min_len]
    )


def accessible_windows(data):
    if np.shape(data)[1] != 2:
        raise ValueError("Data should have two columns: time and signal")
    
    currently_above = False
    intervals = []
    t1 = None
    t2 = None
    for i in range(len(data)):
        t = data[i,0]
        x = data[i,1]
        if currently_above:
            if x < 150:
                t2 = data[i-1,0]
                currently_above = False
                intervals.append([t1,t2])
        else:
            if x >= 150:
                t1 = t
                currently_above = True

    if currently_above:
        t2 = data[-1, 0]
        intervals.append([t1, t2])
    return np.array(intervals)

###############################

def get_max_amplitudes(amplitudes, N):
    """Simply takes the N biggest frequencies"""
    max_indices = np.argpartition(amplitudes, -N)[-N:]
    return max_indices

def get_peaks(amplitudes, N):
    # Find all local maxima
    peaks, _ = find_peaks(amplitudes, height=0, distance=1000)
    # Get the top K from the ACTUAL peaks only
    top_peaks_indices = peaks[np.argsort(amplitudes[peaks])[-N:]]
    return top_peaks_indices


###############################

def model_rmse(model: TideModel, k):
    """Computes the Root Mean Squared Error of the prediction model."""
    true_signal = model.get_total_signal()
    
    assert k <= (model.validation_interval[1] - model.validation_interval[0])
    
    pred_indices = np.linspace(model.validation_interval[0], model.validation_interval[1]-1, k).astype(int)
    pred_times = pred_indices * dt

    prediction = model.predict_array(pred_times)
    true_signal_subsamples = true_signal[pred_indices]
    return rmse(prediction, true_signal_subsamples)

def rmse(arr1, arr2):
    errors = arr1 - arr2
    rmse = np.sqrt(np.mean(errors**2))
    return rmse

##################

def index_to_time(start_time, index):
    return start_time + datetime.timedelta(minutes=index*dt)
def time_to_index(start_time, time):
    delta = time - start_time
    return delta.total_seconds() / (dt * 60)

def tui(model: TideModel):
    print("Type 'exit' to terminate.")
    while True:
        print("\nEnter a timestamp (YYYY-MM-DD hh:mm):")
        inp_text = input()
        if inp_text.lower() == 'exit': break
        
        try:
            time = datetime.strptime(inp_text, "%Y-%m-%d %H:%M")
        except ValueError:
            print("Invalid format. Please use: YYYY-MM-DD hh:mm")
            continue
        if time is None:
            print("Invalid format. Please use: YYYY-MM-DD hh:mm")
            continue
            
        relative_time = time_to_index(model.start_time, time)
        prediction = model.predict_single(relative_time)
        print(f"Prediction = {prediction:.0f}cm, Accessible = {prediction>=150}")



def main():
    print("Reading data...")
    raw_signal, start_time = io.read_data("walsoorden2004-2024.csv")
    print("Building model...")
    model = TideModel(raw_signal, start_time, 0.9, windowing=True)
    print("Window", compare_windows(model))
    print("Diff", model_rmse(model,1000))
    # times = np.arange(model.validation_interval[0], model.validation_interval[1]-1)

    # tui(model)

    # prediction = model.predict_array(np.arange(0, 500, step=1)*dt)
    # plot_comparison(model.get_total_signal()[0:500], prediction)

    # print(f"RMSE: {rmse(model, 10000):.3f}")

# Run script
if __name__ == "__main__":
    main()
