import spacy

# Importing Spacy Language Models
import en_core_web_sm
import en_core_web_trf
import en_core_web_md


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

    def respondTo(self, msg):
        guide_statement = self.nlp("You are a chatbot who wants to help the user answer their question.")
        statement = self.nlp(msg)

        print("...")

        for ent in statement.ents:
            print(ent.label_, ent.text)




