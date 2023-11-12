import config
import telebot
import openai
from prepare_data import load_paragraphs, prepare_data
from search_matches import find_matches

bot = telebot.TeleBot(config.TOKEN)
openai.api_key = config.OPENAI_API_KEY

paragraphs = load_paragraphs()
X, vectorizer, morph, lemmatized_paragraphs = prepare_data(paragraphs)

@bot.message_handler(func=lambda message: True)
def send_matches(message):
    matches = find_matches(
        message.text, X, vectorizer, morph, lemmatized_paragraphs, paragraphs
    )
    user_question = message.text

    # Формирование промпта с вопросом пользователя и совпадениями
    prompt = f"Ты — ассистент по сортировке вторсырья для переработки. Ответь на вопрос пользователя, используя информацию из абзацев ниже, не добавляя свою информацию. Ты можешь только перефразировать данные, чтобы ответзвучал естественнее. Вот вопрос: '{user_question}'. Дай конкретный и полезный ответ, используя данные:\n\n"
    prompt += "\n\n".join([f"{match[0]}" for match in matches])
    prompt += "\n\n Будь внимателен, это очень важно для моей карьеры."
    print(prompt)
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=250,
        temperature=0.3,
        stop=["."],
    )

    # Отправка ответа пользователю
    bot.reply_to(message, response.choices[0].text.strip())


bot.polling()
