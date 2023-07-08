import spacy

import en_core_web_sm
import en_core_web_trf
import en_core_web_md

import nltk as nlkt
import nltk.data
from nltk.sentiment import SentimentIntensityAnalyzer as sia

from nltk.chat.util import Chat, reflections
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.punkt import PunktLanguageVars, PunktParameters, PunktToken


# Warning, by scrolling down, you will see me commit (get it haha) a cardinal sin...
# Commenting lines of code explaining what they do instead of why I did it that way *shock*
# Doing this so that we can have an easier time with the slideshow & explanations


class ChatBotInfo:
    def __init__(self, name, autoresponse):
        self.name = name
        self.autoresponse = autoresponse


words = None
stopwords = None


class ChatBot:

    def __init__(self, language_model, min_similarity, info):
        self.language_model = language_model
        self.nlp = spacy.load(self.language_model)
        self.min_similarity = min_similarity
        self.info = info

        nltk.download([
            "names",
            "stopwords",
            "state_union",
            "twitter_samples",
            "movie_reviews",
            "averaged_perceptron_tagger",
            "vader_lexicon",
            "punkt",
        ])

        # Testing purposes only can safely be deleted later 7/7/2023
        nltk.download('shakespeare')

        global words
        global stopwords

        words = [word for word in nltk.corpus.state_union.words() if word.isalpha()]  # Alphanumeric :D
        stopwords = nltk.corpus.stopwords.words('english')

        pol_scores = sia().polarity_scores("Yes")  # Detecting positive/negative/neutral tone
        print(pol_scores)

        print(words[0:50])

        words = [word for word in words if word.lower() not in stopwords]  # Removed the undesired words!
        # TODO Work on implementing data analysis into a single function. Return large dataset or variables to manage
        #   complexity.

        print(words[0:50])

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
        input_msg = None

        self._get_info("")

        while input_msg != "end":
            input_msg = input("> ")

            print(self._respond_to(input_msg))

            print(self.spacy_tokenize(input_msg))
            print(self.sent_word_tokenize(input_msg))

            print("epic")
            self._get_info(input_msg)
