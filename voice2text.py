import requests
import json
import uuid
import speech_recognition as sr

COG_URL1 = "https://centralindia.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
#COG_URL2 = "https://speech.platform.bing.com/speech/recognition/interactive/cognitiveservices/v1?language=en-IN&locale=en-IN"
COG_URL2 = 'https://centralindia.stt.speech.microsoft.com/speech/recognition/interactive/cognitiveservices/v1?language=en-IN'



def stotext(audio):
    sample_rate = 48000
    # Chunk is like a buffer. It stores 2048 samples (bytes of data)
    # here.
    # it is advisable to use powers of 2 such as 1024 or 2048
    chunk_size = 1024
    # Initialize the recognizer
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio)
        print("you said: " + text)

        # error occurs when google could not understand what was said

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")

    except sr.RequestError as e:
        print("Could not request results from Google".format(e))

    return text

def stream_audio_file(speech_file, chunk_size=1024):
    with open(speech_file, 'rb') as f:
        while 1:
            data = f.read(1024)
            if not data:
                break
            yield data

def retrieve_transcript(file):
    r = requests.post(COG_URL1, headers = {
        'Content-type': 'application/x-www-form-urlencoded',
        'Content-length': '0',
        'Ocp-Apim-Subscription-Key': 'c2192c5518f34c5ba909ad141bdd473b'
        #'Ocp-Apim-Subscription-Key': '71b467489a964dd492e579a0f6c8de9c'
    })

    #print "in retrieve_transcript"
    token = r.text

    #print token

    try:
        d = stream_audio_file(file)
        print("dddddd")
        print(d)
        print("eeeee")

        s = requests.post(COG_URL2, data = stream_audio_file(file), headers = {
                        'Authorization': 'Bearer ' + token,
                        'Content-type': 'audio/wav; codec="audio/pcm"; samplerate=44100'
                        })
    except Exception as e:
        print(e)

    #print s
    #print "printing s"

    #print s.json()

    print("in voice2text")
    print(s.json())
    print("done in voice")
    return s.json()['DisplayText']
                        
        
    
