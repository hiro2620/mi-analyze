import sys
import pickle

if len(sys.argv) < 2:
    print("Usage: python main.py <path_to_pkl_file>")
    sys.exit(1)

pkl_file_path = sys.argv[1]

with open(pkl_file_path, 'rb') as f:
    loaded_data = pickle.load(f)

# 読み込んだデータの取り出し
epochs_data = loaded_data['epochs_data']
labels = loaded_data['labels']