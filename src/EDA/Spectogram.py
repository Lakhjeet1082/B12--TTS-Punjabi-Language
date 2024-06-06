import librosa
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns
from glob import glob
import gc
import os
# from IPython.display import display, Audio

male_data_file_path = glob("/home/sarabjot/Batch_12_punjabi/Dataset/PUNJABI_MALE/wav/*.wav")

for i in range(len(male_data_file_path)):
    plots_done = glob("/home/sarabjot/Batch_12_punjabi/EDA/Spectogram/*.png")
    name = os.path.basename(male_data_file_path[i])
    name = name[:len(name)-4]
    if f"/home/sarabjot/Batch_12_punjabi/EDA/Amplitude/{name}.png" in plots_done:
        continue
    female_data_file_path = f"/home/sarabjot/Batch_12_punjabi/Dataset/PUNJABI_FEMALE/wav/{name}.wav"
    male_amp, male_sr = librosa.load(male_data_file_path[i])
    try:
        female_amp, female_sr = librosa.load(female_data_file_path)
    except:
        continue
    time_in_sec_male = male_amp.shape[0]/male_sr
    time_in_sec_female = female_amp.shape[0]/female_sr

    stft_male = librosa.stft(male_amp)
    stft_female = librosa.stft(female_amp)

    S_db_male = librosa.amplitude_to_db(np.abs(stft_male))
    S_db_female = librosa.amplitude_to_db(np.abs(stft_female))

    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1_plot = librosa.display.specshow(S_db_male, sr=male_sr, x_axis='time', y_axis='log',ax=ax1)
    ax1.set_title('Spectrogram Male')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency (Hz)')
    ax2_plot = librosa.display.specshow(S_db_female, sr=female_sr, x_axis='time', y_axis='log',ax=ax2)
    ax2.set_title('Spectrogram Female')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency (Hz)')
    mel = librosa.feature.melspectrogram(y=male_amp, sr=male_sr)
    ax3_plot = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', x_axis='time',ax=ax3)
    ax3.set_title('Mel Spectrogram Male')
    mel = librosa.feature.melspectrogram(y=female_amp, sr=female_sr)
    ax4_plot = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', x_axis='time',ax=ax4)
    ax4.set_title('Mel Spectrogram Female')
    plt.colorbar(ax1_plot,ax=ax1)
    plt.colorbar(ax2_plot,ax=ax2)
    plt.colorbar(ax3_plot,ax=ax3)
    plt.colorbar(ax4_plot,ax=ax4)

    plt.savefig(f"/home/sarabjot/Batch_12_punjabi/EDA/Spectogram/{name}.png")
    plt.close('all')
    print(f"{name} is done, file number {i}")
    del name, male_amp, male_sr, female_amp, female_sr, time_in_sec_female, time_in_sec_male, stft_male, stft_female, S_db_male,S_db_female
    gc.collect()