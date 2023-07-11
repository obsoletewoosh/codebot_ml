from codebot_ml.KeywordBot.chat_bot_keyword import ChatBot, ChatBotInfo
from codebot_ml.GenerativeBot.generative_chat_bot import GenChatBot


chat_bot = None


def _init(language_model):
    global chat_bot

    info = ChatBotInfo(name="robot man", desc="Nice Robot", train_responses="NA")
    chat_bot = ChatBot(language_model=language_model, min_similarity=0.75, info=info)


if __name__ == '__main__':
    #_init(language_model='en_core_web_md')
    #chat_bot.converse()
    GenChatBot("Joe")
