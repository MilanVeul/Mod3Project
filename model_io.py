import numpy as np

def read_data(file):
    """Reads a given csv file and extracts the indices, timestamps and values."""
    data = np.loadtxt(file, skiprows=1, delimiter=";", dtype=str)
    
    indices = data[:,0].astype(int)
    times = None
    # Parsing time is extremely slow, so only enable it if you are actually use it
    # parse = np.vectorize(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    # times = parse(data[:,1])

    tide = data[:,2].astype(int)
    for i, x in enumerate(tide):
        if x != 999999999: continue
        if i == 0:
            data[i] = data[i+1]
            continue

        next_valid_index = -1
        for j in range(i+1, len(tide)):
            if tide[j] != 999999999:
                next_valid_index = j
                break
        if next_valid_index == -1:
            data[i] = data[i + 1]
            continue

        spacing = next_valid_index - (i-1)
        tide[i] = (tide[next_valid_index]*(1/spacing) + tide[i-1]*(1-(1/spacing)))

    # Remove invalid measurements
    # mask = (tide != 999999999)
    # tide = tide[mask]
    # indices = indices[mask]
    return indices, times, tide

def save_model(model, filename="models/model.npz"):
    w, A, phi, mu = model
    np.savez(filename, omegas=w, amplitudes=A, arguments=phi, mean=mu)
    print(f"Model saved to {filename}")

def load_model(filename="models/model.npz"):
    data = np.load(filename)
    model = (data['omegas'], data['amplitudes'], data['arguments'], data['mean'])
    print(f"Model loaded from {filename}")
    return model