from telegram import Update, Chat
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from model import predict

BOT_TOKEN = "***"


async def check_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message

    if not message or not message.text:
        return

    text = message.text
    label, score = predict(text)

    if (label == "insult") | (label == "dangerous") | (label == "obscenity"):
        response = (
            f"⚠️ Обнаружено агрессивное сообщение\n"
            f"Вероятность токсичности: {score:.2f}\n\n"
            f"❗ Пожалуйста, соблюдайте нормы общения."
        )
        await message.reply_text(response)

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_message))

print("Бот запущен. Готов к проверке сообщений в группах.")
app.run_polling()

