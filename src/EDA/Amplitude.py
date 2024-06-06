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
    plots_done = glob("/home/sarabjot/Batch_12_punjabi/EDA/Amplitude/*.png")
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

    fig = plt.figure(figsize=(50,40))
    axes = fig.add_subplot(211)
    male, = axes.plot(np.linspace(0,time_in_sec_male,male_amp.shape[0]),male_amp,label="Male")
    female, = axes.plot(np.linspace(0,time_in_sec_female,female_amp.shape[0]),female_amp,label="Female")
    axes.set_xlabel("Time",fontsize=50)
    axes.set_ylabel('Amplitude',fontsize=50)
    axes.legend(handles=[male,female])
    axes1 = fig.add_subplot(223)
    male, = axes1.plot(np.linspace(0,time_in_sec_male,male_amp.shape[0]),male_amp,label="Male")
    axes1.set_xlabel("Time",fontsize=50)
    axes1.set_ylabel('Amplitude(Male)',fontsize=50)
    axes2 = fig.add_subplot(224)
    female, = axes2.plot(np.linspace(0,time_in_sec_female,female_amp.shape[0]),female_amp,label="Male")
    axes2.set_xlabel("Time",fontsize=50)
    axes2.set_ylabel('Amplitude(Female)',fontsize=50)
    plt.savefig(f"/home/sarabjot/Batch_12_punjabi/EDA/Amplitude/{name}.png")
    plt.close('all')
    print(f"{name} is done, file number {i}")
    del name, male_amp, male_sr, female_amp, female_sr, time_in_sec_female, time_in_sec_male
    gc.collect()