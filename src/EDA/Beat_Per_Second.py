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
    plots_done = glob("/home/sarabjot/Batch_12_punjabi/EDA/Beat_Per_Second/*.png")
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

    onset_male = librosa.onset.onset_strength(y=male_amp,sr=male_sr)
    onset_female = librosa.onset.onset_strength(y=female_amp,sr=female_sr)
    male_tempo, male_beats = librosa.beat.beat_track(onset_envelope=onset_male,sr=male_sr)
    female_tempo, female_beats = librosa.beat.beat_track(onset_envelope=onset_female,sr=female_sr)
    beat_time_male = librosa.frames_to_time(male_beats,sr=male_sr)
    beat_time_female = librosa.frames_to_time(female_beats,sr=male_sr)
    time_intervals_male = [beat_time_male[i + 1] - beat_time_male[i] for i in range(len(beat_time_male) - 1)]
    beats_per_second_male = [1 / interval for interval in time_intervals_male]
    time_intervals_female = [beat_time_female[i + 1] - beat_time_female[i] for i in range(len(beat_time_female) - 1)]
    beats_per_second_female = [1 / interval for interval in time_intervals_female]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    female = ax.bar(beat_time_female[:-1], beats_per_second_female, width=0.05, color='b', label='Female Beats per Second')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Beats per Second')

    male = ax.bar(beat_time_male[:-1], beats_per_second_male, width=0.05, color='r', label='Male Beats per Second')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Beats per Second')

    plt.title(f'Beats per Second in {name}')
    plt.grid(True)
    plt.legend(handles=[male,female])

    plt.savefig(f"/home/sarabjot/Batch_12_punjabi/EDA/Beat_Per_Second/{name}.png")
    plt.close('all')
    print(f"{name} is done, file number {i}")
    del name, male_amp, male_sr, female_amp, female_sr, time_in_sec_female, time_in_sec_male
    gc.collect()