from telegram import Update
from telegram.ext import ContextTypes
from models.text_model import predict
from models.audio_model import predict_audio_sentiment

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

async def check_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice or update.message.audio
    if not voice:
        return
    file = await voice.get_file()
    audio_bytes = await file.download_as_bytearray()
    response = predict_audio_sentiment(audio_bytes)
    await update.message.reply_text(response)