import numpy as np

def read_data(file):
    """Reads a given csv file and extracts the indices, timestamps and values."""
    data = np.loadtxt(file, skiprows=1, delimiter=";", dtype=str)
    
    indices = data[:,0].astype(int)
    times = None
    # Parsing time is extremely slow, so only enable it if you are actually using it
    # parse = np.vectorize(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    # times = parse(data[:,1])

    tide_raw = data[:,2].astype(int)
    # Remove invalid measurements
    mask = (tide_raw != 999999999)
    tide = tide_raw[mask]

    indices = indices[mask]
    return indices, times, tide

def generate_cosine(N):
    indices = np.arange(N)
    times = None

    A = 5
    f = 1
    phi = 0
    values = A*np.cos(f*indices + phi)
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