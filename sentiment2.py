from __future__ import unicode_literals, print_function
import spacy
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from pycorenlp import StanfordCoreNLP
from sklearn.preprocessing import LabelEncoder

# from pydub import AudioSegment
# import speech_recognition as sr
# import pyaudio

from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import librosa
import glob
from sklearn import preprocessing

def stream_audio_file(speech_file, chunk_size=1024):
    with open(speech_file, 'rb') as f:
        while 1:
            data = f.read(1024)
            if not data:
                break
            yield data


def getSentResult(text):
    nlp = StanfordCoreNLP('http://localhost:9000')
    res = nlp.annotate(text,
                       properties={
                           'annotators': 'sentiment',
                           'outputFormat': 'json',
                           'timeout': 1000,
                       })
    for s in res["sentences"]:
        '''
        print("%d: '%s': %s %s" % (
            s["index"],
            " ".join([t["word"] for t in s["tokens"]]),
            s["sentimentValue"], s["sentiment"]))

        '''
        sentimentVal = s["sentimentValue"]
        sentimentLabel = s["sentiment"]
    return sentimentLabel


def getToneResult(file):
    lb = LabelEncoder()

    model = Sequential()
    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(216, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.load_weights("Emotion_Voice_Detection_Model.h5")
    print("Loaded model from disk")

    livedf1 = pd.DataFrame(columns=['feature'])
    #files = os.listdir(audioDir)
    #print(files)
    toneOutput = []
    negativeWeight = 0
    positiveWeight = 0
    #for i in files:
    #filepath = audioDir + "/" + i
    X, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    featurelive = mfccs
    livedf1 = featurelive
    # print('feature shape:')
    # print(livedf1.shape,livedf1)
    livedf1 = pd.DataFrame(data=livedf1)
    # print(livedf1.shape)
    livedf1 = livedf1.stack().to_frame().T
    livedf1
    twodim = np.expand_dims(livedf1, axis=2)
    # print(twodim.shape)
    livepreds = model.predict(twodim,
                              batch_size=32,
                              verbose=1)
    # print(livepreds)
    livepreds1 = livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
   # print('prediction for file ' + i)
    print(type(liveabc),liveabc)
    if liveabc[0] == 1:
         #positiveWeight = positiveWeight + 1
         toneOutput = 1
    else:
         toneOutput = 0
    #print(positiveWeight)
    #print(negativeWeight)
    #toneOutput.append(positiveWeight)
    #toneOutput.append(liveabc)

    '''
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
        except:
            print("Sorry could not recognize what you said")
  
  
  
    def sentiment(textString):
      '''

    print(toneOutput)
    return toneOutput


# new entity label

def load_NER(nlp):
    labels = ['WEATHER', 'PRODUCT', 'INSURANCE', 'CLAIM', 'TRANSACTION']
    # nlp = spacy.load('en_core_web_sm')

    TRAIN_DATA = [
        ("Sky", {
            'entities': [(0, 3, 'WEATHER')]
        }),

        ("Product", {
            'entities': [(0, 7, 'PRODUCT')]
        }),

        ("insurance", {
            'entities': [(0, 9, 'INSURANCE')]
        }),

        ("covered", {
            'entities': [(0, 7, 'CLAIM')]
        }),

        ("DEDUCTED", {
            'entities': [(0, 8, 'TRANSACTION')]
        }),

        ("BALANCE", {
            'entities': [(0, 6, 'TRANSACTION')]
        })
    ]

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)  # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    for i in labels:
        ner.add_label(i)

    # if model is None:
    #    optimizer = nlp.begin_training()
    # else:
    #    # Note that 'begin_training' initializes the models, so it'll zero out
    #    # existing entity types.
    optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(20):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)
    # nlp.enable_pipes(*other_pipes)
    return nlp


def predict(text, nlp):
    # doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
    # doc = nlp(u'today sky is very beautifulx')
    # doc = nlp(u'today it is very beautiful Sky')
    # doc = nlp(u'Do you think your product is very very good')
    doc = nlp(text)
    dep = ['ROOT', 'nsubj', 'pcomp', 'dobj', 'pobj', 'attr']

    pos = ['NOUN', 'VERB', 'PROPN']

    sum = ''
    sum2 = ''
    cont = ''
    cont2 = ''
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
        if token.pos_ in pos:
            # print(token.text)
            sum2 += token.text + ' '
            if token.dep_ in dep:
                # print(token.text,token.lemma_,token.dep_)
                sum += token.text + ' '

    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
        cont2 += ent.label_ + ' '
        if token.pos_ in pos:
            print(token.text)

            if token.dep_ in dep:
                cont += ent.label_ + ' '
    return sum, cont, cont2, sum2
    #    #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

    # for chunk in doc.noun_chunks:
    #    print(chunk.text, chunk.root.text, chunk.root.dep_,
    #          chunk.root.head.text)


def main(text):
    nlp = spacy.load('en_core_web_sm')


    # text = input('Enter text:')
    sum, cont, cont2, sum2 = predict(text, nlp)
    print(sum)
    print(sum2)
    print(cont)
    print(cont2)
    print('')
    nlp = load_NER(nlp)
    sum, cont, cont2, sum2 = predict(text, nlp)
    print(sum)
    print(sum2)
    print(cont)
    print(cont2)


    return sum2,cont2


if __name__ == '__main__':
    plac.call(main)
