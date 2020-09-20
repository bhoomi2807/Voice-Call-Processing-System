import pyaudio
import wave
import flask, flask.views
from flask import Flask, render_template, request, redirect, url_for, flash, make_response, session
import os
from os import path
import voice2text
import sentiment


# import main
# from main import flag

flag = 0

import keyboard



# Code from: https://gist.github.com/mabdrabo/8678538

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 500
audio = pyaudio.PyAudio()
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
frames = []



def stream_audio_file(speech_file, chunk_size=1024):
    with open(speech_file, 'rb') as f:
        while 1:
            data = f.read(1024)
            if not data:
                break
            yield data

def write_audio(frames, index):
    dirname="sad2"
    file_name = dirname+"/s"+str(index)+".wav"
    waveFile = wave.open(dirname+"/s"+str(index)+".wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    # calling toneAnalysis
    tone_result = sentiment.getToneResult(file_name)
    with open('./templates/toneData.csv','a') as f:
        print("in file write")
        f.write(str(tone_result) + ",")
        f.close()
    print("toneAnalysis: " +str(tone_result))






def record(s):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    print("recording...")
    #print session['stopRecord']
    with open('flag.txt', 'w') as f:
        f.write("1")

    frame2=[]
    i=0;
    index=0;
    while True:
        if not path.exists("flag.txt"):# if key 'q' is pressed
            print('You stop button!')
            break  # finishing the loo
        data = stream.read(CHUNK)
        frames.append(data)
        frame2.append(data)
        i+=1
        if i == int(RATE/CHUNK * 4):
            write_audio(frame2, index)
            index+=1
            i = 0
            frame2=[]



    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open("karma.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    spoken_text = voice2text.retrieve_transcript("karma.wav")
    #spoken_text = voice2text.stotext(stream_audio_file("karma.wav"))

    print("spoken text")
    print(spoken_text)
    print("spoken text done")

    #print "done"

    #   list_words = [a.lower().strip("!,.?") for a in spoken_text.split()]
    sent_result = sentiment.getSentResult(spoken_text)
    with open('./templates/sentiment.txt','w') as f:
        print("in file write")
        f.write(sent_result)
        f.close()


    print("sentiment_result " + sent_result)

    summary,context = sentiment.main(spoken_text)

    with open('./templates/spoken_text.txt','w') as f:
        print("in file write")
        f.write(spoken_text)
        f.close()

    with open('./templates/context.txt','w') as f:
        print("in file write")
        f.write("summary: "+ summary + "\r\ncontext: " + context )
        f.close()

    print("summary" + summary)
    print("context" + context)

    #print list_words
    #print spoken_text
    return spoken_text



# generic error processing
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(CHUNK)
    #     frames.append(data)
    # print("finished recording")
    # stop()




def stop():
    #print "stop"
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open("karma.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()






