from telegram import Update, ChatPermissions
from telegram.ext import ContextTypes
from bot.config import chat_levels
from models.text_model import predict
from models.audio_model import predict_audio_sentiment
import datetime



async def is_admin(update: Update) -> bool:
    user_id = update.effective_user.id
    chat_admins = await update.effective_chat.get_administrators()
    return any(admin.user.id == user_id for admin in chat_admins)


async def set_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update):
        await update.message.reply_text("⛔ Только администратор может менять уровень реакции.")
        return

    try:
        level = int(context.args[0])
        if level not in [1, 2, 3, 4]:
            raise ValueError
        chat_id = update.effective_chat.id
        chat_levels[chat_id] = level
        await update.message.reply_text(f"✅ Уровень токсичности установлен на {level}")
    except (IndexError, ValueError):
        await update.message.reply_text("❗ Использование: /setlevel [1|2|3|4]")


async def warn_user(update: Update, score: float):
    await update.message.reply_text(
        f"⚠️ Обнаружено агрессивное сообщение\n"
        f"Вероятность токсичности: {score:.2f}\n\n"
        f"❗ Пожалуйста, соблюдайте нормы общения."
    )


async def mute_user(update: Update):
    user = update.effective_user
    chat = update.effective_chat

    until_date = datetime.datetime.utcnow() + datetime.timedelta(seconds=60)

    try:
        await chat.restrict_member(
            user.id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until_date
        )
        await update.message.reply_text("⛔ Пользователь временно замьючен на 1 минуту.")
    except Exception as e:
        await update.message.reply_text(f"❗ Не удалось замьютить пользователя: {e}")


async def ban_user(update: Update):
    chat = update.effective_chat
    user = update.effective_user

    try:
        await chat.ban_member(user.id)
        await chat.unban_member(user.id)

        await update.message.reply_text(f"🚫 Пользователь {user.full_name} был удалён из чата за токсичное поведение.")
    except Exception as e:
        await update.message.reply_text(f"❗ Не удалось удалить пользователя: {e}")


async def process_text(update: Update, text: str):
    label, score = predict(text)

    level = chat_levels[update.effective_chat.id]

    if (label == "insult") | (label == "dangerous") | (label == "obscenity"):
        if level == 1:
            await warn_user(update, score)
        elif level == 2:
            await warn_user(update, score)
            await mute_user(update)
        elif level == 3:
            await ban_user(update)


async def check_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not message or not message.text:
        return

    await process_text(update, message.text)


async def check_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice or update.message.audio
    if not voice:
        return

    file = await voice.get_file()
    audio_bytes = await file.download_as_bytearray()
    response, score = predict_audio_sentiment(audio_bytes)
    print(response, score)
    level = chat_levels[update.effective_chat.id]

    if "негатив" in response:
        if level == 1:
            await warn_user(update, score)
        elif level == 2:
            await warn_user(update, score)
            await mute_user(update)
        elif level == 3:
            await ban_user(update)
