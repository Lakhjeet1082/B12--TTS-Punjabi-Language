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
    plots_done = glob("/home/sarabjot/Batch_12_punjabi/EDA/MFCC/*.png")
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

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    mfcc_male = librosa.feature.mfcc(y=male_amp, sr=male_sr)
    mfcc_female = librosa.feature.mfcc(y=female_amp, sr=female_sr)
    male = librosa.display.specshow(mfcc_male, x_axis='time', cmap='coolwarm', ax = ax)
    female = librosa.display.specshow(mfcc_female, x_axis='time', cmap='coolwarm',ax = ax1)
    plt.colorbar(male,ax=ax)
    plt.colorbar(female,ax=ax1)
    ax.set_title('MFCC Heatmap for Male')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MFCC Coefficient')
    ax1.set_title('MFCC Heatmap for Female')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('MFCC Coefficient')

    plt.savefig(f"/home/sarabjot/Batch_12_punjabi/EDA/MFCC/{name}.png")
    plt.close('all')
    print(f"{name} is done, file number {i}")
    del name, male_amp, male_sr, female_amp, female_sr, time_in_sec_female, time_in_sec_male, mfcc_female, mfcc_male
    gc.collect()