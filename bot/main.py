from telegram.ext import ApplicationBuilder, MessageHandler, filters
from bot.handlers import check_message, check_audio


BOT_TOKEN = "***"

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_message))
app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, check_audio))

print("Бот запущен. Готов к проверке сообщений.")
app.run_polling()