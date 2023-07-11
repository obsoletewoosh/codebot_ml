import os

import nltk as nlkt
from nltk.stem.lancaster import LancasterStemmer as ls
import numpy as np
import tflearn
import json
import pickle
import random

import spacy

import nltk.data
from nltk.sentiment import SentimentIntensityAnalyzer as sia

from sklearn.model_selection import train_test_split

# Warning, by scrolling down, you will see me commit (get it haha) a cardinal sin...
# Commenting lines of code explaining what they do instead of why I did it that way *shock*
# Doing this so that we can have an easier time with the slideshow & explanations

# TODO Make it so that generated files go into the datafolder and are named correctly if possible...

os.chdir("KeywordBot")

with open('intents.json') as intents:
    data = json.load(intents)


class ChatBotInfo:
    def __init__(self, name, desc, train_responses):
        self.name = name
        self.desc = desc
        self.train_responses = train_responses


class ChatBot:

    def _init_model(self):
        try:
            with open('../data.pickle', 'rb') as f:
                words, labels, training, output = pickle.load(f)
        except:
            # Fetching and Feeding information--
            words = []
            labels = []
            x_docs = []
            y_docs = []

            for intent in data['intents']:
                for pattern in intent['patterns']:
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    x_docs.append(wrds)
                    y_docs.append(intent['tag'])


                    if intent['tag'] not in labels:
                        labels.append(intent['tag'])

            words = [ls().stem(w.lower()) for w in words if w not in "?"]
            words = sorted(list(set(words)))
            labels = sorted(labels)

            training = []
            output = []

            out_empty = [0 for _ in range(len(labels))]

            # One hot encoding, Converting the words to numerals
            for x, doc in enumerate(x_docs):
                bag = []
                wrds = [ls().stem(w) for w in doc]
                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:]
                output_row[labels.index(y_docs[x])] = 1

                training.append(bag)
                output.append(output_row)

            training = np.array(training)
            output = np.array(output)

            with open('../data.pickle', 'wb') as f:
                pickle.dump((words, labels, training, output), f)

        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
        net = tflearn.regression(net)

        model = tflearn.DNN(net)

        try:
            model.load(self.info.name.strip(" ")+"model.tflearn")
        except:
            model = tflearn.DNN(net)
            model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            model.save(self.info.name.strip(" ")+'model.tflearn')

        return model, words, labels

    def __init__(self, language_model, min_similarity, info):
        self.s_words = None
        self.bag = None

        self.language_model = language_model
        self.nlp = spacy.load(self.language_model)
        self.min_similarity = min_similarity
        self.info = info

        self.model, self.words, self.labels = self._init_model()
        print(self.model)

    def word_bag(self, s, words):
        self.bag = [0 for _ in range(len(words))]
        self.s_words = [ls().stem(word.lower()) for word in nltk.word_tokenize(s)]

        for se in self.s_words:
            for i, w in enumerate(words):
                if w == se:
                    self.bag[i] = 1

        return np.array(self.bag)

    def spacy_tokenize(self, msg):
        return [token for token in self.nlp(msg)]

    def sent_word_tokenize(self, msg):
        sent = nltk.sent_tokenize(msg)
        word = nltk.word_tokenize(msg)

        return sent, word

    def _get_info(self, msg):
        sentences = nlkt.sent_tokenize(msg)

        emotion_list = [sia().polarity_scores(i) for i in sentences]

        pos = 0
        neu = 0
        neg = 0

        for i in emotion_list:
            pos += i['pos']
            neu += i['neu']
            neg += i['neg']

        pos /= len(emotion_list)
        neu /= len(emotion_list)
        neg /= len(emotion_list)

        overall_tone = {'neg': neg, 'neu': neu, 'pos': pos}
        print(overall_tone)

    def _respond_to(self, msg):
        guide_statement = self.nlp("?")
        statement = self.nlp(msg)

        return "N/A"

    def converse(self):
        global data

        while True:
            response_msg = self.info.name + ": "
            input_msg = input("> ")
            for msg in nltk.sent_tokenize(input_msg):
                if msg == "end":
                    break

                results = self.model.predict([self.word_bag(msg, self.words)])

                results_index = np.argmax(results)

                tag = self.labels[results_index]

                for i in data['intents']:

                    if i['tag'] == tag:
                        responses = i['responses']
                        response_sent = random.choice(responses)
                        print(response_sent)
                        response_msg += response_sent + " "
            response_msg.format('bot_name', self.info.name)
            print(response_msg)
