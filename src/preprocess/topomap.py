import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

source = Path("data/") / "20250512" / "processed" / "ishii0005_ica_epochs.pkl"

# pickleのEEGデータを読み込む
with open(source, 'rb') as f:
    raw = pickle.load(f)

epochs  = raw['epochs']
labels = raw['labels']

# epochs[0].average().animate_topomap(times=np.arange(epochs.tmin, epochs.tmax, 0.1), ch_type='eeg', show=True, frame_rate=10, blit=False)
epochs[0].average().animate_topomap(times=np.arange(1.0, 1.1, 0.002), ch_type='eeg', show=True, frame_rate=30, blit=False)
plt.show()