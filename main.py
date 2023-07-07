import spacy

# Importing Spacy Language Models
import en_core_web_sm
import en_core_web_trf
import en_core_web_md

from chat_bot import ChatBot

# Definine Global variables with None so that they can be defined in __init__ to manage complexity
nlp = None


def _init(language_model):
    global nlp

    # Model difference based upon efficient boolean since one is faster than the other
    # The slower model is more accurate so there is a tradeoff which needs to be specified at runtime

    if language_model == 'en_core_web_sm':
        nlp = spacy.load(language_model)
        nlp = en_core_web_sm.load()
    elif language_model == 'en_core_web_md':
        nlp = spacy.load(language_model)
        nlp = en_core_web_md.load()


def to_token_list(doc):
    return [token for token in doc]


if __name__ == '__main__':
    # _init(language_model='en_core_web_md')

    # All warnings related to 'nlp' not being callable should be ignored
    # nlp is defined during runtime

    # doc = nlp("Some sentance here")
    # tokens = to_token_list(doc)
    # print(tokens)

    a = ChatBot(language_model='en_core_web_md', min_similarity=0.50)
    a.respondTo("I really want to go to New York... Or Rhode Island! The school named Brown is really interesting! Where is Brown and is it nice?")

    # doc1 = nlp(input("Insert sentence one!"))
    # doc2 = nlp(input("Insert sentence two!"))

    # similarity = doc1.similarity(doc2)

    # print(f"The similarity between sentence one and two are: {similarity * 100:.0f}%")
