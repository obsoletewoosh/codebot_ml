import spacy
import en_core_web_sm
import en_core_web_trf

nlp = None
efficient_mode = None

def __init__(efficient):
    global nlp
    global efficient_mode

    efficient_mode = efficient

    if efficient:
        nlp = spacy.load('en_core_web_sm')
        nlp = en_core_web_sm.load()
    else:
        nlp = spacy.load('en_core_web_trf')
        nlp = en_core_web_trf.load()


def to_token_list(doc):
    return [token for token in doc]


if __name__ == '__main__':
    __init__(efficient=True)

    doc = nlp("Some sentance here")
    tokens = to_token_list(doc)
    print(tokens)

    doc1 = nlp(input("Insert sentence one!"))
    doc2 = nlp(input("Insert sentence two!"))

    similarity = doc1.similarity(doc2)

    print(f"The similarity between sentence one and two are: {similarity*100:.0f}%")

    #Ok

