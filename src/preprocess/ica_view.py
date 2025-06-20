import csv
import mne
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import pickle
from pathlib import Path


file_path = "data/20250604/20250604_1_ishii.vhdr"
# .vhdrファイルを指定してEEGデータを読み込む
raw = mne.io.read_raw_brainvision(file_path, preload=True)
# rename_mapping = {ch: ELECTRODE_NAMES[i] for i, ch in enumerate(raw.ch_names)}
# raw.rename_channels(rename_mapping)
raw.set_montage('easycap-M1')

raw.filter(l_freq=1, h_freq=40)
fig = raw.compute_psd(tmax=np.inf, fmax=70).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)

# ICAによるノイズ除去
print("ICAによるノイズ除去を開始します...")
ica = mne.preprocessing.ICA(n_components=20, random_state=42)
ica.fit(raw)
# fig = ica.plot_components()

fig = ica.plot_sources(raw)

plt.show()