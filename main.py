import spacy
import en_core_web_sm

nlp = spacy.load('en_core_web_sm')

def _init():
    nlp = en_core_web_sm.load()
    doc = nlp("Hello World! See how parts of the sentence are separated into different groups!")

    print([(w.text, w.pos_) for w in doc])


if __name__ == '__main__':
    _init()
