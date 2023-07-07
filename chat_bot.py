import spacy

import en_core_web_sm
import en_core_web_trf
import en_core_web_md

import nltk
from nltk.chat.util import Chat, reflections


class ChatBot:

    def __init__(self, language_model, min_similarity):
        self.language_model = language_model
        self.nlp = spacy.load(self.language_model)
        self.min_similarity = min_similarity

    def __str__(self):
        return str(self.language_model)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.language_model == other.language_model
        return False

    def respond_to(self, msg):
        guide_statement = self.nlp("?")
        statement = self.nlp(msg)

        # TODO Make the respond_to function work correctly!

        return "N/A"
