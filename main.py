import record
import voice2text
import os
from os import path
import flask, flask.views, flask.sessions
from flask import Flask, render_template, request, redirect, url_for, flash, make_response, session

from flask import Markup
from flask import jsonify
import requests

import shutil
import os
import time
from flask import Flask
import record
from record import flag
import sentiment

flag = 0

# from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.urandom(24)
spoken_text = ""
list_words = ""


class Main(flask.views.MethodView):
    def get(self):
        return flask.render_template('index.html')


class About(flask.views.MethodView):
    def get(self):
        return flask.render_template('about.html')


class Contact(flask.views.MethodView):
    def get(self):
        return flask.render_template('contact.html')


class Test(flask.views.MethodView):
    def get(self):
        return flask.render_template('test.html')


class Admin(flask.views.MethodView):
    def get(self):
        return flask.render_template('index2.html')


class ToneData(flask.views.MethodView):
    def get(self):
        return flask.render_template('toneData.csv')


class Context(flask.views.MethodView):
    def get(self):
        return flask.render_template('context.txt')


class Sentiment(flask.views.MethodView):
    def get(self):
        return flask.render_template('sentiment.txt')


class Text(flask.views.MethodView):
    def get(self):
        return flask.render_template('spoken_text.txt')




# CORS(app)

app.add_url_rule('/', view_func=Main.as_view('karma'), methods=["GET"])
app.add_url_rule('/about/', view_func=About.as_view('about'), methods=["GET"])
app.add_url_rule('/contact/', view_func=Contact.as_view('contact'), methods=["GET"])
app.add_url_rule('/test/', view_func=Test.as_view('test'), methods=["GET"])
app.add_url_rule('/admin/', view_func=Admin.as_view('admin'), methods=["GET"])
app.add_url_rule('/toneData/', view_func=ToneData.as_view('toneData'), methods=["GET"])
app.add_url_rule('/sentiment/', view_func=Sentiment.as_view('sentiment'), methods=["GET"])
app.add_url_rule('/context/', view_func=Context.as_view('context'), methods=["GET"])

app.add_url_rule('/admin/', view_func=Admin.as_view('chart'), methods=["GET"])
app.add_url_rule('/spoken_text/', view_func=Text.as_view('text'), methods=["GET"])




@app.route("/start_record")
def rec():
    session['stopRecord'] = 0
    file_name = "temp.wav"


    with open('./templates/toneData.csv', 'w') as f:
        f.close()

    # if path.exists("./templates/sentiment.txt"):
    #     os.remove("./templates/sentiment.txt")
    # if path.exists("./templates/context.txt"):
    #     os.remove("./templates/context.txt")

    return record.record("file_name")


@app.route("/stop_record")
def stop():
    global flag
    flag = 1
    # record.stop()
    # return convert_text()


# not calling now
@app.route("/convert_text")
def convert_text():
    global spoken_text, list_words
    spoken_text = voice2text.retrieve_transcript("karma.wav")

    print("done")

    list_words = [a.lower().strip("!,.?") for a in spoken_text.split()]

    print(list_words)
    print(spoken_text)

    return spoken_text


# not calling
@app.route("/karma")
def main():
    file_name = "temp.wav"
    # shutil.rmtree("data/")
    # os.mkdir("data")

    record.record(file_name)

    spoken_text = voice2text.retrieve_transcript("temp.wav")

    # print "done"

    list_words = [a.lower().strip("!,.?") for a in spoken_text.split()]

    # print list_words
    # print spoken_text
    return spoken_text


@app.route('/change_status')
def change_status():
    # print "in change status"
    session['stopRecord'] = 1
    session.modified = True
    if (path.exists("flag.txt")):
        os.remove("flag.txt")
    # print session['stopRecord']
    return "change Status"


# main()

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=6789)
