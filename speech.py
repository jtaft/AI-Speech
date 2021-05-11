# This is an attempt at a Machine Learning Speech to Text Converter
#
#
# Audio Files are saved in a folder structure:
#   /test-data/<collection-name>/<audio-files>
#   /test-data/<collection-name>/audio-data.json
#
# audio-data.json is Formatted:
#   { <filename (no .wav)>: <content as text>,
#     ... }
#

import os

os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"

import keras
import librosa
import numpy as np
import json
from keras import models
from keras import layers
from keras.utils import to_categorical

TEST_DATA_FOLDER_PATH = '/home/jay/karas/voice-data-service/test-data/'
FIRST_SAMPLE_PATH = TEST_DATA_FOLDER_PATH + 'first-sample/'
VALIDATION_SAMPLE_PATH = TEST_DATA_FOLDER_PATH + 'validation-sample/'
AUDIO_JSON = 'audio-data.json'

def get_vectors_from_path(path):
  #read json dict
  with open(path + AUDIO_JSON) as audio_json:
    audio_dict = json.load(audio_json)


    directory = os.fsencode(path)
    mfcc_vectors = []
    labels = []
    for file in os.listdir(directory):
      mfcc = None
      filename = os.fsdecode(file)
      if filename.endswith(".wav"):
        #append vectors for audio files
        wave, sr = librosa.load(path + filename, mono=True, sr=None)
        mfcc = librosa.feature.mfcc(wave, sr=sr)
        #print(len(wave))
        print(mfcc.shape)
        #print(len(mfcc[0]))
        #print(len(mfcc[1]))
        if (40 > mfcc.shape[1]):
          mfcc = np.pad(mfcc, pad_width=((0,0),(0,40-mfcc.shape[1])), mode='constant')
          #print(mfcc.shape)
        else:
          mfcc = mfcc[:, :40]
        mfcc_vectors.append(mfcc)


        #convert yes/no to binary and append to label list
        filename_no_extension = filename.replace(".wav", "")
        labels.append(int(audio_dict[filename_no_extension].lower() == "yes" ))
  #print("labels: " + str(len(labels[0])))
  #print("vectors: " + str(len(mfcc_vectors[0])))

  return np.array(labels, dtype='float32'), np.array(mfcc_vectors)

sample_labels, sample_audio_vectors = get_vectors_from_path(FIRST_SAMPLE_PATH)
test_labels, test_audio_vectors = get_vectors_from_path(VALIDATION_SAMPLE_PATH)

#for i in sample_audio_vectors:
#    print(i.shape)

#print(sample_labels.shape)
#print(sample_audio_vectors.shape)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(20,40)))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

#print(sample_audio_vectors.shape)
#print(sample_labels.shape)
history = model.fit(sample_audio_vectors, sample_labels, epochs=1, batch_size=52)
results = model.evaluate(test_audio_vectors, test_labels)

print(model.metrics_names)
print(results)
