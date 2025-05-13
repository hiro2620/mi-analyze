import mne
import sys
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def visualize_eeg(file_path):
    """
    EEGデータを読み込み、可視化する関数
    
    Parameters:
    -----------
    file_path : str
        .vhdrファイルのパス
    """
    # .vhdrファイルを指定してEEGデータを読み込む
    raw = mne.io.read_raw_brainvision(file_path, preload=True)
    
    # データの基本情報を表示
    print(f"サンプリング周波数: {raw.info['sfreq']} Hz")
    print(f"計測時間: {raw.times.max():.2f} 秒")
    print(f"チャンネル数: {len(raw.ch_names)}")
    print(f"チャンネル名: {raw.ch_names}")
    
    # データをフィルタリング (0.5-40Hz)
    raw.filter(0.5, 40)
    
    # データをプロット
    raw.plot(scalings='auto', title='EEGデータ可視化', show=True)
    
    # スペクトログラムも表示
    
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    # パワースペクトル密度
    spectrum = raw.compute_psd(fmin=0,fmax=60)
    spectrum.plot(show=True)

    plt.show()

if __name__ == "__main__":
    # .vhdrファイルを指定 (.eegデータの実体は.vhdrファイルから参照される)
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_vhdr_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    visualize_eeg(file_path)