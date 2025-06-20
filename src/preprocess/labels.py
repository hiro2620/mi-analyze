import mne

file_path = "data/20250604/20250604_1_ishii.vhdr"

# .vhdrファイルを指定してEEGデータを読み込む
raw = mne.io.read_raw_brainvision(file_path, preload=True)
print(len(raw.ch_names))
print(raw.ch_names)
print("FCz" in raw.ch_names)