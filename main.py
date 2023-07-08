import spacy
import en_core_web_sm
import en_core_web_trf
import en_core_web_md

from chat_bot import ChatBot, ChatBotInfo

chat_bot = None


def _init(language_model):
    global chat_bot

    info = ChatBotInfo(name="Mr. Robot", autoresponse="Hello!")
    chat_bot = ChatBot(language_model=language_model, min_similarity=0.75, info=info)


if __name__ == '__main__':
    _init(language_model='en_core_web_md')
    chat_bot.converse()
