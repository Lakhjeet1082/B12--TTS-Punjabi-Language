import phonemizer 
import os
import glob
import gc
import time
input_path = glob.glob("/home/sarabjot/batch_12_punjabi/Acoustic_Model/oovs_found_pro_dict.txt")
language = 'pa'
backend = 'espeak'
file1 = open("pro_dict_new.txt",mode='r')
word_done = file1.read().split("\n")
file1.close()
file1 = open("pro_dict_new.txt",mode='a')
#file2 = open("Already_Done.txt",mode='a')
for i in range(len(input_path)):
    try:
        if input_path[i] not in done:
            with open(input_path[i]) as file:
                line = file.readline()
                words = line.split(" ")
                for word in words:
                    ph = phonemizer.phonemize(word, language=language, backend=backend)
                    new_str = ""
                    for w in ph:
                        new_str += w + " "
                    insert = word + "\t" + new_str
                    if insert not in word_done:
                        # print(word + " Done")
                        word_done.append(insert)
                        file1.write(insert+'\n')
                        file1.flush()
                    del ph, insert 
                    gc.collect()
                file.close()
                #file2.write(input_path[i]+"\n")
                #file2.flush()
                #done.append(input_path[i])
        print("-----------------------------------------")
        print(f"{i} is done")
        print("-----------------------------------------")
    except Exception as e:
        print(f"{os.path.basename(input_path[i])} has some error {e}")
        continue