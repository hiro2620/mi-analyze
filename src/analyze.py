import sys
import pickle
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python main.py <path_to_pkl_file>")
    sys.exit(1)

pkl_file_path = sys.argv[1]

with open(pkl_file_path, 'rb') as f:
    loaded_data = pickle.load(f)

# 読み込んだデータの取り出し
epochs_data = loaded_data['epochs_data']
labels = loaded_data['labels']

print(f"{epochs_data.shape=}")
print(f"{labels}")

# FFTの実行
n_epochs, n_channels, n_times = epochs_data.shape
print(f"データ構造: {n_epochs}エポック × {n_channels}チャンネル × {n_times}時点")

# FFT結果を格納する配列
fft_results = np.zeros((n_epochs, n_channels, n_times//2 + 1), dtype=complex)

# 各エポック、各チャンネルに対してFFTを実行
for epoch_idx in range(n_epochs):
    for channel_idx in range(n_channels):
        # 現在のエポックとチャンネルの時系列データを取得
        time_series = epochs_data[epoch_idx, channel_idx, :]
        
        # FFTを実行
        fft_result = fft.rfft(time_series)
        
        # 結果を格納
        fft_results[epoch_idx, channel_idx, :] = fft_result

print(f"FFT結果の形状: {fft_results.shape}")

# FFT結果の絶対値（パワースペクトル）を計算
power_spectra = np.abs(fft_results)**2

# サンプリング周波数の推定（必要に応じて適切な値に変更）
# 一般的なEEGのサンプリング周波数は250Hzや500Hz程度
sampling_rate = 500  # Hz（仮の値）

# 周波数ビンの計算
freq_bins = np.fft.rfftfreq(n_times, d=1/sampling_rate)

# 結果の可視化（例: 最初のエポックの最初のチャンネル）
plt.figure(figsize=(10, 6))
plt.plot(freq_bins, power_spectra[0, 0, :])
plt.title('パワースペクトル (エポック 0, チャンネル 0)')
plt.xlabel('周波数 (Hz)')
plt.ylabel('パワー')
plt.xlim([0, 50])  # 一般的なEEG解析では50Hz以下が多い
plt.savefig('first_epoch_channel_spectrum.png')
print("最初のエポックとチャンネルのスペクトルを保存しました。")

# 全チャンネルの平均パワースペクトル（エポック0）
plt.figure(figsize=(10, 6))
plt.plot(freq_bins, np.mean(power_spectra[0], axis=0))
plt.title('平均パワースペクトル (エポック 0, 全チャンネル)')
plt.xlabel('周波数 (Hz)')
plt.ylabel('パワー')
plt.xlim([0, 50])
plt.savefig('first_epoch_average_spectrum.png')
print("最初のエポックの平均スペクトルを保存しました。")

# 保存するかどうか
save_result = input("FFT結果を保存しますか？ (y/n): ")
if save_result.lower() == 'y':
    with open('fft_results.pkl', 'wb') as f:
        pickle.dump({
            'fft_results': fft_results,
            'power_spectra': power_spectra,
            'freq_bins': freq_bins,
            'labels': labels
        }, f)
    print("FFT結果を保存しました。")