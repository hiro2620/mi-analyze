import sys
import csv
import mne
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import pickle

MI_TASK_DURATION_MS = 3000
ELECTRODE_NAMES = [
    'C2', 'Cz', 'C1', 'C3', 'CP2', 'CPz', 'CP1', 'CP3',
    'TP9', 'O1', 'P7', 'C5', 'Fp1', 'F7', 'FC3', 'T7',
    'CP6', 'C4', 'C6', 'T8', 'F8', 'Fp2', 'FC6', 'AFz',
    'FC1', 'FCz', 'FC4', 'FC2', 'CP4', 'TP10', 'O2', 'P8',
]

# .vhdrファイルを指定 (.eegデータの実体は.vhdrファイルから参照される)
if len(sys.argv) < 3:
    print("Usage: python main.py <path_to_vhdr_file> <path_to_seq_file>")
    sys.exit(1)

file_path = sys.argv[1]
seq_file_path = sys.argv[2]


task_sequence = []
with open(seq_file_path, 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # ヘッダー行をスキップ
    for row in csv_reader:
        if len(row) >= 2:  # 2列目が存在することを確認
            try:
                # 2列目を整数として抽出
                task_sequence.append(int(row[1]))
            except (ValueError, IndexError):
                # 整数に変換できない場合やインデックスエラーの場合はスキップ
                print(f"Invalid data in row: {row}")
                continue

# .vhdrファイルを指定してEEGデータを読み込む
raw = mne.io.read_raw_brainvision(file_path, preload=True)
rename_mapping = {ch: ELECTRODE_NAMES[i] for i, ch in enumerate(raw.ch_names)}
raw.rename_channels(rename_mapping)
raw.set_montage('easycap-M1')

# remove drift
raw.filter(l_freq=1.0, h_freq=40)

# ICAによるノイズ除去
print("ICAによるノイズ除去を開始します...")
ica = mne.preprocessing.ICA(n_components=20, random_state=42)
ica.fit(raw)


ica.plot_sources(raw)
ica.exclude = [0, 2, 15, 14, 19]
# ica.exclude = [i for i in range(20)]
ica.plot_properties(raw, picks=ica.exclude, psd_args={'fmax': 50}, show=False)

# ノイズ除去を適用
raw_cleaned = raw.copy()
ica.apply(raw_cleaned)

# 元のデータと比較
raw.plot(duration=5, n_channels=20, title='Original Data')
raw_cleaned.plot(duration=5, n_channels=20, title='ICA Cleaned Data')

# 以降の処理では、raw_cleanedを使用
raw = raw_cleaned

raw_events, _labels = mne.events_from_annotations(raw)
triggers_mask = raw_events[:, 2] == 254
event_timestamps = raw_events[triggers_mask, 0]

assert len(task_sequence) == len(event_timestamps), "Task sequenceとevent timestampsの長さが一致しません"
events = []
for i, t in enumerate(event_timestamps):
    events.append([t, 0, task_sequence[i]])
    events.append([t+MI_TASK_DURATION_MS*raw.info['sfreq']/1000.0, 0, 9999])

events = np.array(events, dtype=int)

mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, show=True, event_id=None)

# (エポック数, チャンネル数, 時間サンプル数)
epochs = mne.Epochs(raw, events, tmin=-1, tmax=2, preload=True)
# epochs = epochs["1"]
epochs.plot(scalings="auto", events=events)
plt.show()


# ファイルの保存先を設定
output_file_path = file_path.replace('.vhdr', '_epochs.pkl')
epochs_list = epochs.get_data()
labels = epochs.events[:,-1]

# データを保存するための辞書を作成
data_to_save = {
    'epochs_data': epochs_list,
    'labels': labels,
    'sfreq': raw.info['sfreq'],
    'ch_names': raw.info['ch_names']
}

# pickleを使用してデータを保存
# with open(output_file_path, 'wb') as f:
#     pickle.dump(data_to_save, f)

print(f"データを {output_file_path} に保存しました")