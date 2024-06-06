import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
from glob import glob
import os

def chunks(audio_file_path):
    myaudio = AudioSegment.from_file(audio_file_path, format="wav")

    chunk_length_ms = 30000

    chunks = make_chunks(myaudio, chunk_length_ms)
    

def transcribe_long_audio(audio_file_path):

    recognizer = sr.Recognizer()

    audio = AudioSegment.from_wav(audio_file_path)

    chunk_length_ms = 30000

    chunks = make_chunks(audio, chunk_length_ms)
    transcribed_text = ""
    i = 1
    print("--------------------------------------------------------")
    for chunk in chunks:
            audio_data = sr.AudioData(chunk.raw_data,sample_rate = chunk.frame_rate, sample_width=chunk.sample_width)
            try:
                text = recognizer.recognize_google(audio_data,language = "pa")
                transcribed_text += text + " "
                print(f"{i} iteration done")
                i+=1
            except sr.UnknownValueError:
                pass
    print("--------------------------------------------------------")     
    return transcribed_text

if __name__ == "__main__":
    all_files = glob("/home/sarabjot/batch_12_punjabi/iitm_data/Audio_male/*.wav")
    for i in range(len(all_files)): 
        basename = os.path.basename(all_files[i])
        basename = basename[:len(basename)-4]
        audio_file_path = all_files[i]
        result = transcribe_long_audio(audio_file_path)
        with open(f"/home/sarabjot/batch_12_punjabi/iitm_data/transcriptions/{basename}.lab",mode = "w") as f:
             f.write(result)
             f.flush()
             f.close()
        print(f"{i} is done")