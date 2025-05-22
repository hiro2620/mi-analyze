import csv
import pickle
import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

data_file_path = "data/20250512/processed/ishii0008_ica.pkl"
seq_file_path = "data/tasks/hand2/task-sequence.csv"

MI_TASK_DURATION_MS = 3000

# pickleのEEGデータを読み込む
with open(data_file_path, 'rb') as f:
    raw = pickle.load(f)

# タスク順序を読み込む
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

raw_events, _labels = mne.events_from_annotations(raw)
triggers_mask = raw_events[:, 2] == 254
event_timestamps = raw_events[triggers_mask, 0]
print(f"event_timestamps: {event_timestamps}")

assert len(task_sequence) == len(event_timestamps), "Task sequenceとevent timestampsの長さが一致しません"
events = []
for i, t in enumerate(event_timestamps):
    events.append([t, 0, task_sequence[i]])
    events.append([t+MI_TASK_DURATION_MS*raw.info['sfreq']/1000.0, 0, 9999])

events = np.array(events, dtype=int)

# mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, show=True, event_id=None)

# (エポック数, チャンネル数, 時間サンプル数)
epochs = mne.Epochs(raw, events, tmin=-1, tmax=2, preload=True)

# インタラクティブにエポックを表示してbadエポックをマーキング
# (プロット上でクリックして不良エポックを選択できます)
fig = epochs.plot(scalings="auto", events=events)
plt.show()

# badとしてマークされたエポックの情報を表示
n_dropped = len([log for log in epochs.drop_log if len(log) > 0])
print(f"badとしてマークされたエポック数: {n_dropped}")
print(f"残りのエポック数: {len(epochs)}")


# ファイルの保存先を設定
output_file_name= Path(data_file_path).stem + "_epochs.pkl"
save_dir = Path(data_file_path).parent
output_file_path = save_dir / output_file_name
epochs_list = epochs.get_data()
labels = epochs.events[:,-1]

print(epochs_list.shape)

# データを保存するための辞書を作成
data_to_save = {
    'epochs_data': epochs_list,
    'labels': labels,
    'sfreq': raw.info['sfreq'],
    'ch_names': raw.info['ch_names'],
    'drop_log': epochs.drop_log,  # 不良とマークされたエポックの記録
    'selection': epochs.selection,  # 保持されているエポックのインデックス
    'drop_indices': np.where(np.array([len(log) > 0 for log in epochs.drop_log]))[0]  # 除外されたエポックのインデックス
}

# pickleを使用してデータを保存
with open(output_file_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"データを {output_file_path} に保存しました")