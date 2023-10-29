import config
import telebot
from prepare_data import load_paragraphs, prepare_data
from search_matches import find_matches

bot = telebot.TeleBot(config.TOKEN)

paragraphs = load_paragraphs()
X, vectorizer, morph, lemmatized_paragraphs = prepare_data(paragraphs)

@bot.message_handler(func=lambda message: True)
def send_matches(message):
    matches = find_matches(message.text, X, vectorizer, morph, lemmatized_paragraphs, paragraphs)
    response = "\n\n".join([f"{i+1}. {match[0]}" for i, match in enumerate(matches)])
    bot.reply_to(message, response)

bot.polling()
