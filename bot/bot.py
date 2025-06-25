import os
import numpy as np
import librosa
import tensorflow as tf
from tempfile import NamedTemporaryFile
from keras.models import load_model
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

from models.text_model import predict

BOT_TOKEN = "***"
AUDIO_MODEL_PATH = '../models/audio_sentiment_model.h5'

SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 13
MAX_PAD_LEN = 174

audio_model = load_model(AUDIO_MODEL_PATH)


def preprocess_audio_bytes(audio_bytes):
    with NamedTemporaryFile(delete=False, suffix=".ogg") as temp:
        temp.write(audio_bytes)
        temp_path = temp.name

    try:
        signal, sr = librosa.load(temp_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
        if mfccs.shape[1] < MAX_PAD_LEN:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, MAX_PAD_LEN - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        return mfccs[np.newaxis, ..., np.newaxis]
    except Exception as e:
        print(f"Ошибка обработки аудио: {e}")
        return None
    finally:
        try:
            os.unlink(temp_path)
        except Exception as e:
            print(f"Ошибка удаления временного файла: {e}")


def predict_audio_sentiment(audio_bytes):
    processed = preprocess_audio_bytes(audio_bytes)
    if processed is None:
        return "⚠️ Не удалось обработать аудиосообщение"
    prediction = audio_model.predict(processed)
    negative_prob = prediction[0][1]
    if negative_prob > 0.5:
        return f"⚠️ Обнаружено негативное аудиосообщение\nВероятность негативности: {negative_prob:.2f}"
    else:
        return f"✅ Аудио не содержит негативного контента (вероятность: {1 - negative_prob:.2f})"


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

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_message))
app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, check_audio))

print("Бот запущен. Готов к проверке сообщений в группах.")
app.run_polling()

