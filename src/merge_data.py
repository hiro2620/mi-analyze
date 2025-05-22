import pickle
from pathlib import Path
import numpy as np

base_path = Path("data/20250512/processed")
entries = [
    "ishii0005_ica_epochs.pkl",
    "ishii0006_ica_epochs.pkl",
    "ishii0007_ica_epochs.pkl",
    "ishii0008_ica_epochs.pkl",
]

output_path = base_path / "ishii_task2_merged.pkl"

# すべてのpickleファイルを読み込み、データをマージする
all_epochs = []
all_labels = []
for entry in entries:
    file_path = base_path / entry
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        all_epochs.append(data['epochs'].get_data())
        all_labels.append(data['labels'])

# axis=0の方向に結合する
merged_epochs = np.concatenate(all_epochs, axis=0)
merged_labels = np.concatenate(all_labels, axis=0)

# マージしたデータを保存する
merged_data = {
    'epochs_data': merged_epochs,
    'labels': merged_labels
}

with open(output_path, 'wb') as f:
    pickle.dump(merged_data, f)

print(f"マージデータを保存しました: {output_path}")
print(f"Epochs shape: {merged_epochs.shape}")
print(f"Labels shape: {merged_labels.shape}")