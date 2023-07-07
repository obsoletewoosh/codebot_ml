import spacy

# Importing Spacy Language Models
import en_core_web_sm
import en_core_web_trf
import en_core_web_md

# Definine Global variables with None so that they can be defined in __init__ to manage complexity
nlp = None
efficient_mode = None


def __init__(efficient):
    global nlp
    global efficient_mode

    efficient_mode = efficient

    # Model difference based upon efficient boolean since one is faster than the other
    # The slower model is more accurate so there is a tradeoff which needs to be specified at runtime

    if efficient:
        nlp = spacy.load('en_core_web_sm')
        nlp = en_core_web_sm.load()
    else:
        nlp = spacy.load('en_core_web_md')
        nlp = en_core_web_md.load()


def to_token_list(doc):
    return [token for token in doc]

def 

if __name__ == '__main__':
    __init__(efficient=True)

    # All warnings related to 'nlp' not being callable should be ignored
    # nlp is defined during runtime

    doc = nlp("Some sentance here")
    tokens = to_token_list(doc)
    print(tokens)

    doc1 = nlp(input("Insert sentence one!"))
    doc2 = nlp(input("Insert sentence two!"))

    similarity = doc1.similarity(doc2)

    print(f"The similarity between sentence one and two are: {similarity * 100:.0f}%")
