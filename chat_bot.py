import spacy

import en_core_web_sm
import en_core_web_trf
import en_core_web_md

import nltk as nlkt
import nltk.data

from nltk.chat.util import Chat, reflections
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.punkt import PunktLanguageVars, PunktParameters, PunktToken


class ChatBot:

    def __init__(self, language_model, min_similarity):
        self.language_model = language_model
        self.nlp = spacy.load(self.language_model)
        self.min_similarity = min_similarity

        nlkt.download('punkt')

    def __str__(self):
        return str(self.language_model)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.language_model == other.language_model
        return False

    def spacy_tokenize(self, msg):
        return [token for token in self.nlp(msg)]

    def sent_word_tokenize(self, msg):
        sent = nltk.sent_tokenize(msg)
        word = nltk.word_tokenize(msg)

        return sent, word

    def _respond_to(self, msg):
        guide_statement = self.nlp("?")
        statement = self.nlp(msg)

        return "N/A"

    def converse(self):
        input_msg = None

        while input_msg != "end":
            input_msg = input("> ")

            print(self._respond_to(input_msg))

            print(self.spacy_tokenize(input_msg))
            print(self.sent_word_tokenize(input_msg))
