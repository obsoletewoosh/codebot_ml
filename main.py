import spacy
import en_core_web_sm

nlp = spacy.load('en_core_web_sm')


def __init__():
    nlp = en_core_web_sm.load()
    doc = nlp("Hello World! See how parts of the sentence are separated into different groups!")

    for token in doc:
        print(f"""
        TEXT  : {token.text} -- Raw text of the word
        LEMMA : {token.lemma_} -- Base form of the token, with no inflectional suffixes.
        POS_  : {token.pos_} -- Coarse-grained part-of-speech from the Universal POS tag set (String)
        POS   : {token.pos} -- Same as POS_ but instead has an integer value
        TAG_  : {token.tag_} -- Fine-grained part-of-speech.
        TAG   : {token.tag} -- Same as TAG_ but instead returns an integer value
        DEP_  : {token.dep_} -- Syntactic dependency relation
        DEP   : {token.dep} -- Returns the integer value of DEP 
        SHAPE : {token.shape_} -- Transform of the token’s string to show orthographic features X for alpha d for numeric
        ALPHA : {token.is_alpha} 
        STOP  : {token.is_stop}
        """)

    print([(w.text, w.pos_, w.pos) for w in doc])

if __name__ == '__main__':
    __init__()
