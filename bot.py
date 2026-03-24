import asyncio
import logging
import os
from openai import OpenAI
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery,
    LabeledPrice, PreCheckoutQuery, ReplyKeyboardMarkup, KeyboardButton,
)
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# ─── Загрузка .env ────────────────────────────────────────────────────────────
load_dotenv()

TELEGRAM_TOKEN       = os.getenv("TELEGRAM_TOKEN")
NVIDIA_API_KEY       = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL         = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
ADMIN_USERNAME       = os.getenv("ADMIN_USERNAME", "@your_username")
ADMIN_ID             = int(os.getenv("ADMIN_ID", 0))  # твой Telegram user_id
STARS_PRICE          = int(os.getenv("STARS_PRICE", 100))  # цена в Stars
FREE_LIMIT           = int(os.getenv("FREE_LIMIT", 5))
MAX_HISTORY          = int(os.getenv("MAX_HISTORY", 20))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", 800))  # Снижено для экономии
TEMPERATURE          = float(os.getenv("TEMPERATURE", 0.7))
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS", "firebase.json")

if not TELEGRAM_TOKEN or not NVIDIA_API_KEY:
    raise ValueError("Заполни TELEGRAM_TOKEN и NVIDIA_API_KEY в файле .env")

# ─── Firebase Firestore ───────────────────────────────────────────────────────
cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred)
db = firestore.client()
# Коллекции:
#   users/{user_id}  → { messages_used, paid, model, username }
#   history/{user_id}/messages/{doc_id} → { role, content, ts }
# ─────────────────────────────────────────────────────────────────────────────

# ─── Доступные модели по тарифам ─────────────────────────────────────────────
# Free: только базовые модели
FREE_MODELS: dict[str, str] = {
    "🌬️ Mistral 7B":      "mistralai/mistral-7b-instruct-v0.3",
    "💎 GLM-4.7":         "z-ai/glm4_7",
}

# Basic: все модели кроме топовых
BASIC_MODELS: dict[str, str] = {
    "🌬️ Mistral 7B":      "mistralai/mistral-7b-instruct-v0.3",
    "💎 GLM-4.7":         "z-ai/glm4_7",
    "🦙 Llama 3.1 70B":   "meta/llama-3.1-70b-instruct",
    "🔥 MiniMax M2.1":    "minimaxai/minimax-m2_1",
}

# Pro: все модели
PRO_MODELS: dict[str, str] = {
    "🦙 Llama 3.1 70B":   "meta/llama-3.1-70b-instruct",
    "🦙 Llama 3.3 70B":   "meta/llama-3.3-70b-instruct",
    "⚡ Nemotron 70B":     "nvidia/llama-3.1-nemotron-70b-instruct",
    "🌬️ Mistral 7B":      "mistralai/mistral-7b-instruct-v0.3",
    "🐉 Qwen 3.5 122B":   "qwen/qwen3.5-122b-a10b",
    "💎 GLM-4.7":         "z-ai/glm4_7",
    "🔥 MiniMax M2.1":    "minimaxai/minimax-m2_1",
}

# Тарифы
TIERS = {
    "free":  {"name": "Free",  "limit": 5,   "models": FREE_MODELS,  "price": 0},
    "basic": {"name": "Basic", "limit": 100, "models": BASIC_MODELS, "price": 100},
    "pro":   {"name": "Pro",   "limit": -1,  "models": PRO_MODELS,   "price": 200},  # -1 = безлимит
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Режимы работы бота ──────────────────────────────────────────────────────
MODES = {
    "📚 Домашка": "Ты помогаешь с учебой, решаешь задачи и объясняешь просто. Используй эмодзи для структуры. Без символов ** и markdown.",
    "💸 Заработок": "Ты даешь идеи заработка и конкретные шаги. Используй эмодзи для структуры. Без символов ** и markdown.",
    "✍️ Тексты": "Ты профессиональный копирайтер, пишешь тексты. Используй эмодзи для структуры. Без символов ** и markdown.",
    "🎬 TikTok идеи": "Ты создаешь вирусные идеи и сценарии для TikTok. Используй эмодзи для структуры. Без символов ** и markdown.",
}

user_modes = {}  # {user_id: system_prompt}
user_last_request = {}  # {user_id: timestamp} — защита от спама

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Ты создаёшь ответы для Telegram-бота.\n\n"
        "Формат:\n"
        "— Без символов ** и markdown-разметки\n"
        "— Используй эмодзи для структуры\n"
        "— Делай короткие абзацы\n"
        "— Используй заголовки через эмодзи (не через **)\n\n"
        "Стиль:\n"
        "— Чётко, красиво, читаемо\n"
        "— Без лишней воды\n"
        "— Разбивай на блоки\n\n"
        "Пример структуры:\n\n"
        "🎬 Название: ...\n\n"
        "⏱ Длительность: ...\n\n"
        "📜 Сценарий:\n"
        "1. ...\n"
        "2. ...\n\n"
        "🎵 Музыка: ...\n\n"
        "✨ Эффекты: ...\n\n"
        "#никогда не используй символы ** или Markdown"
    ),
}

# NVIDIA NIM клиент
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)


# ─── Firebase: работа с пользователями ───────────────────────────────────────

def _user_ref(user_id: int):
    return db.collection("users").document(str(user_id))


def _history_ref(user_id: int):
    return db.collection("history").document(str(user_id)).collection("messages")


async def get_user(user_id: int) -> dict:
    """Получить данные пользователя из Firestore (или создать дефолтные)."""
    loop = asyncio.get_event_loop()
    doc = await loop.run_in_executor(None, lambda: _user_ref(user_id).get())
    if doc.exists:
        return doc.to_dict()
    # Новый пользователь — создаём запись
    import datetime
    data = {
        "messages_used": 0,
        "paid": False,
        "tier": "free",
        "model": NVIDIA_MODEL,
        "username": "",
        "last_reset": datetime.datetime.now(datetime.timezone.utc),
    }
    await loop.run_in_executor(None, lambda: _user_ref(user_id).set(data))
    return data


async def update_user(user_id: int, data: dict) -> None:
    """Обновить поля пользователя в Firestore."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: _user_ref(user_id).update(data))


async def get_history(user_id: int) -> list[dict]:
    """Загрузить историю диалога из Firestore (последние MAX_HISTORY сообщений)."""
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(
        None,
        lambda: _history_ref(user_id)
            .order_by("ts", direction=firestore.Query.DESCENDING)
            .limit(MAX_HISTORY)
            .get()
    )
    # Разворачиваем обратно — от старых к новым
    messages = [{"role": d.get("role"), "content": d.get("content")} for d in docs]
    return list(reversed(messages))


async def append_history(user_id: int, role: str, content: str) -> None:
    """Добавить сообщение в историю диалога."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: _history_ref(user_id).add({
            "role": role,
            "content": content,
            "ts": firestore.SERVER_TIMESTAMP,
        })
    )


async def clear_history(user_id: int) -> None:
    """Удалить всю историю диалога пользователя."""
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, lambda: _history_ref(user_id).get())
    for doc in docs:
        await loop.run_in_executor(None, doc.reference.delete)


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def format_answer(text: str) -> str:
    """Пост-обработка ответа: убираем markdown, добавляем красивое форматирование."""
    # Убираем markdown
    text = text.replace("**", "")
    
    # Добавляем отступы между блоками (если AI забыл)
    replacements = {
        "Название:": "\n🎬 Название:",
        "Длительность:": "\n⏱ Длительность:",
        "Сценарий:": "\n\n📜 Сценарий:\n",
        "Музыка:": "\n\n🎵 Музыка:\n",
        "Эффекты:": "\n\n✨ Эффекты:\n",
        "Идеи:": "\n\n💡 Идеи:\n",
        "Хэштеги:": "\n\n#️⃣ Хэштеги:\n",
        "Навыки:": "\n\n🛠 Навыки:\n",
    }
    
    for old, new in replacements.items():
        if old in text and new.strip() not in text:
            text = text.replace(old, new)
    
    return text.strip()


def split_text(text: str, max_length: int = 4000) -> list[str]:
    """Разбивает длинный текст на части по max_length символов."""
    if len(text) <= max_length:
        return [text]
    
    parts = []
    while text:
        if len(text) <= max_length:
            parts.append(text)
            break
        
        # Ищем последний перенос строки в пределах max_length
        split_pos = text.rfind('\n', 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        
        parts.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    
    return parts


def remaining_from(user_data: dict) -> int:
    tier = user_data.get("tier", "free")
    limit = TIERS[tier]["limit"]
    if limit == -1:  # безлимит
        return 999
    return max(0, limit - user_data.get("messages_used", 0))


def has_access_from(user_data: dict) -> bool:
    tier = user_data.get("tier", "free")
    limit = TIERS[tier]["limit"]
    if limit == -1:  # безлимит
        return True
    return user_data.get("messages_used", 0) < limit


def get_available_models(user_data: dict) -> dict[str, str]:
    """Возвращает доступные модели для тарифа пользователя."""
    tier = user_data.get("tier", "free")
    return TIERS[tier]["models"]


def check_model_access(user_data: dict, model_id: str) -> bool:
    """Проверяет доступ к модели для тарифа пользователя."""
    available = get_available_models(user_data)
    return model_id in available.values()


async def check_and_reset_limits(user_id: int, user_data: dict) -> dict:
    """Проверяет и сбрасывает лимиты раз в месяц (только для basic/pro)."""
    import datetime
    tier = user_data.get("tier", "free")
    if tier == "free":
        return user_data  # free не сбрасывается

    last_reset = user_data.get("last_reset")
    if not last_reset:
        return user_data

    now = datetime.datetime.now(datetime.timezone.utc)
    # Если прошёл месяц — сбрасываем
    if (now - last_reset).days >= 30:
        await update_user(user_id, {
            "messages_used": 0,
            "last_reset": now,
        })
        user_data["messages_used"] = 0
        user_data["last_reset"] = now

    return user_data


async def ask_nvidia(user_id: int, user_text: str) -> str:
    """Запрос к NVIDIA NIM API с историей из Firestore."""
    # Сохраняем сообщение пользователя
    await append_history(user_id, "user", user_text)

    # Загружаем историю для контекста
    history = await get_history(user_id)

    # Получаем модель пользователя
    user_data = await get_user(user_id)
    model = user_data.get("model", NVIDIA_MODEL)

    # Получаем system prompt (режим работы или дефолтный)
    if user_id in user_modes:
        system_prompt = {"role": "system", "content": user_modes[user_id]}
    else:
        system_prompt = SYSTEM_PROMPT

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: nvidia_client.chat.completions.create(
            model=model,
            messages=[system_prompt] + history,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        ),
    )

    answer = response.choices[0].message.content.strip()
    await append_history(user_id, "assistant", answer)
    return answer


# ─── Клавиатуры ──────────────────────────────────────────────────────────────

def buy_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"⭐ Открыть безлимит ({STARS_PRICE} Stars)", callback_data="buy")],
    ])


def models_keyboard(user_data: dict) -> InlineKeyboardMarkup:
    available = get_available_models(user_data)
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=f"model:{mid}")]
        for label, mid in available.items()
    ])


def get_main_menu() -> ReplyKeyboardMarkup:
    """Главное меню с режимами работы."""
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [KeyboardButton(text="📚 Домашка"), KeyboardButton(text="💸 Заработок")],
        [KeyboardButton(text="✍️ Тексты"), KeyboardButton(text="🎬 TikTok идеи")],
        [KeyboardButton(text="📊 Статус"), KeyboardButton(text="🧹 Очистить")],
    ])
    return kb


def get_quick_start_menu() -> ReplyKeyboardMarkup:
    """Быстрый старт с готовыми сценариями."""
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [KeyboardButton(text="✍️ Написать текст"), KeyboardButton(text="📚 Сделать домашку")],
        [KeyboardButton(text="💸 Идея заработка"), KeyboardButton(text="🎬 Сценарий TikTok")],
        [KeyboardButton(text="📊 Статус"), KeyboardButton(text="🧹 Очистить")],
    ])
    return kb


# ─── Бот и диспетчер ─────────────────────────────────────────────────────────
bot = Bot(token=TELEGRAM_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    user_id   = message.from_user.id
    user_data = await get_user(user_id)
    user_data = await check_and_reset_limits(user_id, user_data)

    # Сохраняем username при первом старте
    if not user_data.get("username"):
        await update_user(user_id, {"username": message.from_user.username or ""})

    tier = user_data.get("tier", "free")
    tier_name = TIERS[tier]["name"]
    left = remaining_from(user_data)

    if tier == "pro":
        access_line = "✅ У тебя Pro — безлимит"
    else:
        limit = TIERS[tier]["limit"]
        access_line = f"🆓 Бесплатно: {left} сообщений"

    await message.answer(
        "🤖 AskNeuro AI — нейросеть, которая делает за тебя\n\n"
        "Я могу за 10 секунд:\n"
        "✍️ написать сочинение или пост\n"
        "📚 решить домашку и объяснить\n"
        "💸 дать идею заработка с планом\n"
        "🎬 придумать сценарий для TikTok\n\n"
        "👇 Выбери, что нужно сделать:",
        reply_markup=get_quick_start_menu(),
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "🤖 NVIDIA NIM API — выбирай модель под свой тариф.\n\n"
        "📦 Тарифы:\n"
        "• Free: 5 сообщений, 2 базовые модели\n"
        "• Basic: 100 сообщений/месяц, 4 модели (100 ⭐)\n"
        "• Pro: безлимит, все 7 моделей (200 ⭐)\n\n"
        "Используй /model для выбора модели."
    )


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    await clear_history(message.from_user.id)
    await message.answer("🗑 История очищена.", reply_markup=get_main_menu())


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    user_id   = message.from_user.id
    user_data = await get_user(user_id)
    user_data = await check_and_reset_limits(user_id, user_data)
    
    tier = user_data.get("tier", "free")
    tier_name = TIERS[tier]["name"]
    model = user_data.get("model", NVIDIA_MODEL)

    if tier == "pro":
        await message.answer(f"✅ Тариф: Pro (безлимит)\nМодель: `{model}`", parse_mode="Markdown")
    else:
        left = remaining_from(user_data)
        limit = TIERS[tier]["limit"]
        status_text = (
            f"📦 Тариф: {tier_name}\n"
            f"📊 Использовано: {user_data.get('messages_used', 0)}/{limit}\n"
            f"Осталось: {left}\nМодель: `{model}`"
        )
        
        if left == 0:
            status_text += "\n\n🚀 Открой безлимит, чтобы продолжить"
        
        await message.answer(
            status_text,
            parse_mode="Markdown",
            reply_markup=buy_keyboard() if left == 0 else None,
        )


@dp.message(Command("model"))
async def cmd_model(message: Message) -> None:
    user_data = await get_user(message.from_user.id)
    tier = user_data.get("tier", "free")
    await message.answer(
        f"🤖 Текущая модель: `{user_data.get('model', NVIDIA_MODEL)}`\n"
        f"📦 Тариф: {TIERS[tier]['name']}\n\nВыбери новую:",
        parse_mode="Markdown",
        reply_markup=models_keyboard(user_data),
    )


@dp.callback_query(F.data.startswith("model:"))
async def callback_model(call: CallbackQuery) -> None:
    model_id = call.data.split("model:", 1)[1]
    user_data = await get_user(call.from_user.id)
    
    # Проверка доступа к модели
    if not check_model_access(user_data, model_id):
        tier = user_data.get("tier", "free")
        await call.answer(f"❌ Эта модель недоступна на тарифе {TIERS[tier]['name']}", show_alert=True)
        return
    
    await update_user(call.from_user.id, {"model": model_id})
    available = get_available_models(user_data)
    label = next((k for k, v in available.items() if v == model_id), model_id)
    await call.message.answer(f"✅ Модель: {label}\n`{model_id}`", parse_mode="Markdown")
    await call.answer()


@dp.callback_query(F.data == "buy")
async def callback_buy(call: CallbackQuery) -> None:
    """Отправляем инвойс Telegram Stars при нажатии кнопки."""
    await bot.send_invoice(
        chat_id=call.from_user.id,
        title="Доступ к AskNeuro AI",
        description=f"Безлимитный доступ к нейросети — {STARS_PRICE} ⭐",
        payload="ai_access_stars",
        provider_token="",        # пусто — это Telegram Stars (XTR)
        currency="XTR",
        prices=[LabeledPrice(label="Безлимитный доступ", amount=STARS_PRICE)],
    )
    await call.answer()


@dp.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery) -> None:
    """Подтверждаем оплату — всегда ok."""
    await query.answer(ok=True)


@dp.message(F.successful_payment)
async def successful_payment(message: Message) -> None:
    """Выдаём тариф после успешной оплаты Stars."""
    user_id = message.from_user.id
    stars = message.successful_payment.total_amount

    # Определяем тариф по сумме
    if stars >= 200:
        tier = "pro"
    elif stars >= 100:
        tier = "basic"
    else:
        tier = "free"

    import datetime
    await update_user(user_id, {
        "tier": tier,
        "messages_used": 0,
        "last_reset": datetime.datetime.now(datetime.timezone.utc),
    })

    tier_name = TIERS[tier]["name"]
    limit = TIERS[tier]["limit"]
    limit_text = "безлимит" if limit == -1 else f"{limit} сообщений/месяц"

    success_message = (
        f"🎉 Оплата {stars} ⭐ прошла успешно!\n\n"
        f"✅ Тариф {tier_name} активирован.\n"
        f"📊 Лимит: {limit_text}\n"
        "Пиши — я отвечу 🚀"
    )
    
    # Upsell для Basic → Pro
    if tier == "basic":
        success_message += (
            "\n\n━━━━━━━━━━━━━━━\n"
            "🔥 Хочешь ещё мощнее?\n\n"
            "Pro доступ (200 ⭐):\n"
            "— безлимитные ответы\n"
            "— доступ ко всем 7 моделям\n"
            "— быстрее обработка\n\n"
            "Апгрейд всего за +100 ⭐"
        )
    
    await message.answer(success_message)

    # Уведомление админу
    if ADMIN_ID:
        username = message.from_user.username or str(user_id)
        await bot.send_message(
            ADMIN_ID,
            f"💸 Новая оплата!\n"
            f"👤 @{username} (id: {user_id})\n"
            f"⭐ Stars: {stars}\n"
            f"📦 Тариф: {tier_name}"
        )


@dp.message(F.text.lower().in_({"доступ", "купить", "buy", "оплата"}))
async def buy_access(message: Message) -> None:
    await message.answer(
        f"💸 Доступ — {STARS_PRICE} ⭐ Telegram Stars\n\nНажми кнопку для оплаты:",
        reply_markup=buy_keyboard(),
    )


# ─── Обработчики кнопок меню ─────────────────────────────────────────────────

@dp.message(F.text == "✍️ Написать текст")
async def quick_write_text(message: Message) -> None:
    """Быстрый старт: написать текст."""
    user_modes[message.from_user.id] = MODES["✍️ Тексты"]
    await message.answer(
        "✍️ Напиши тему — я создам текст\n\nНапример:\n— \"пост про путешествия\"\n— \"сочинение про экологию\"",
        reply_markup=get_main_menu(),
    )


@dp.message(F.text == "📚 Сделать домашку")
async def quick_homework(message: Message) -> None:
    """Быстрый старт: домашка."""
    user_modes[message.from_user.id] = MODES["📚 Домашка"]
    await message.answer(
        "📚 Напиши задачу — я решу и объясню\n\nНапример:\n— \"реши уравнение x² + 5x + 6 = 0\"\n— \"объясни фотосинтез\"",
        reply_markup=get_main_menu(),
    )


@dp.message(F.text == "💸 Идея заработка")
async def quick_money(message: Message) -> None:
    """Быстрый старт: заработок."""
    user_modes[message.from_user.id] = MODES["💸 Заработок"]
    await message.answer(
        "💸 Напиши, сколько хочешь зарабатывать — я дам план\n\nНапример:\n— \"идея заработка с 0€\"\n— \"как заработать 500€/месяц\"",
        reply_markup=get_main_menu(),
    )


@dp.message(F.text == "🎬 Сценарий TikTok")
async def quick_tiktok(message: Message) -> None:
    """Быстрый старт: TikTok."""
    user_modes[message.from_user.id] = MODES["🎬 TikTok идеи"]
    await message.answer(
        "🎬 Напиши тему — я создам вирусный сценарий\n\nНапример:\n— \"сценарий про животных\"\n— \"идея для танца\"",
        reply_markup=get_main_menu(),
    )


@dp.message(F.text.in_(list(MODES.keys())))
async def set_mode(message: Message) -> None:
    """Установка режима работы бота."""
    user_id = message.from_user.id
    mode_text = message.text
    user_modes[user_id] = MODES[mode_text]
    
    await message.answer(
        f"✅ Режим выбран: {mode_text}\n\nТеперь напиши свой запрос 👇",
        reply_markup=get_main_menu(),
    )


@dp.message(F.text == "📊 Статус")
async def menu_status(message: Message) -> None:
    """Кнопка Статус из меню."""
    await cmd_status(message)


@dp.message(F.text == "🧹 Очистить")
async def menu_clear(message: Message) -> None:
    """Кнопка Очистить из меню."""
    await clear_history(message.from_user.id)
    await message.answer("🗑 История очищена.", reply_markup=get_main_menu())


@dp.message(Command("grant"))
async def cmd_grant(message: Message) -> None:
    """Выдать платный доступ: /grant <user_id>"""
    if message.from_user.id != ADMIN_ID:
        return
    args = message.text.split()
    if len(args) < 2:
        await message.answer("Использование: /grant <user_id>")
        return
    try:
        target_id = int(args[1])
        await update_user(target_id, {"paid": True})
        await message.answer(f"✅ Доступ выдан: {target_id}")
        await bot.send_message(target_id, "🎉 Доступ активирован! Лимитов больше нет.")
    except ValueError:
        await message.answer("❌ Неверный user_id")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@dp.message(Command("admin"))
async def cmd_admin(message: Message) -> None:
    """Статистика бота — только для админа."""
    if message.from_user.id != ADMIN_ID:
        return
    loop = asyncio.get_event_loop()
    try:
        users_docs = await loop.run_in_executor(
            None, lambda: db.collection("users").get()
        )
        users = [d.to_dict() for d in users_docs]
        total   = len(users)
        paid    = sum(1 for u in users if u.get("paid"))
        blocked = sum(1 for u in users if u.get("banned"))
        msgs    = sum(u.get("messages_used", 0) for u in users)

        await message.answer(
            "📊 Статистика бота\n\n"
            f"👥 Всего пользователей: {total}\n"
            f"💸 Платных: {paid}\n"
            f"🚫 Заблокированных: {blocked}\n"
            f"💬 Сообщений отправлено: {msgs}\n\n"
            "Команды:\n"
            "/grant <id>     — выдать доступ\n"
            "/ban <id>       — заблокировать\n"
            "/unban <id>     — разблокировать\n"
            "/broadcast <текст> — рассылка всем"
        )
    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@dp.message(Command("ban"))
async def cmd_ban(message: Message) -> None:
    """Заблокировать пользователя: /ban <user_id>"""
    if message.from_user.id != ADMIN_ID:
        return
    args = message.text.split()
    if len(args) < 2:
        await message.answer("Использование: /ban <user_id>")
        return
    try:
        target_id = int(args[1])
        await update_user(target_id, {"banned": True})
        await message.answer(f"🚫 Пользователь {target_id} заблокирован.")
        await bot.send_message(target_id, "🚫 Ваш аккаунт заблокирован.")
    except ValueError:
        await message.answer("❌ Неверный user_id")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@dp.message(Command("unban"))
async def cmd_unban(message: Message) -> None:
    """Разблокировать пользователя: /unban <user_id>"""
    if message.from_user.id != ADMIN_ID:
        return
    args = message.text.split()
    if len(args) < 2:
        await message.answer("Использование: /unban <user_id>")
        return
    try:
        target_id = int(args[1])
        await update_user(target_id, {"banned": False})
        await message.answer(f"✅ Пользователь {target_id} разблокирован.")
        await bot.send_message(target_id, "✅ Ваш аккаунт разблокирован.")
    except ValueError:
        await message.answer("❌ Неверный user_id")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")


@dp.message(Command("broadcast"))
async def cmd_broadcast(message: Message) -> None:
    """Рассылка всем пользователям: /broadcast <текст>"""
    if message.from_user.id != ADMIN_ID:
        return
    text = message.text.removeprefix("/broadcast").strip()
    if not text:
        await message.answer("Использование: /broadcast <текст>")
        return

    loop = asyncio.get_event_loop()
    users_docs = await loop.run_in_executor(
        None, lambda: db.collection("users").get()
    )

    sent, failed = 0, 0
    for doc in users_docs:
        uid = int(doc.id)
        if uid == ADMIN_ID:
            continue
        try:
            await bot.send_message(uid, f"📢 {text}")
            sent += 1
        except Exception:
            failed += 1
        await asyncio.sleep(0.05)  # защита от flood

    await message.answer(f"📢 Рассылка завершена.\n✅ Отправлено: {sent}\n❌ Ошибок: {failed}")


@dp.message(F.text)
async def handle_message(message: Message) -> None:
    user_id   = message.from_user.id
    user_data = await get_user(user_id)
    user_data = await check_and_reset_limits(user_id, user_data)

    # Проверка бана
    if user_data.get("banned"):
        await message.answer("🚫 Ваш аккаунт заблокирован.")
        return

    # Защита от спама (макс 1 запрос в 3 секунды)
    import time
    current_time = time.time()
    last_request = user_last_request.get(user_id, 0)
    if current_time - last_request < 3:
        await message.answer("⏳ Подожди немного, я ещё обрабатываю предыдущий запрос...")
        return
    user_last_request[user_id] = current_time

    if not has_access_from(user_data):
        tier = user_data.get("tier", "free")
        limit = TIERS[tier]["limit"]
        await message.answer(
            "🔒 Бесплатные сообщения закончились\n\n"
            "🚀 Что ты получишь:\n"
            "— безлимитные ответы\n"
            "— быстрый AI без очередей\n"
            "— доступ ко всем режимам\n"
            "— помощь с любыми задачами\n\n"
            "💸 Разблокируй доступ и продолжи 👇",
            reply_markup=buy_keyboard(),
        )
        return

    # ВАУ-эффект для первого сообщения
    if user_data.get("messages_used", 0) == 0:
        await message.answer("⚡ Сейчас покажу, что я умею...")

    # Счётчик (для всех кроме pro)
    tier = user_data.get("tier", "free")
    if tier != "pro":
        new_count = user_data.get("messages_used", 0) + 1
        await update_user(user_id, {"messages_used": new_count})
        left = max(0, TIERS[tier]["limit"] - new_count)
        if left == 1:
            await message.answer(
                "⚠️ Остался последний бесплатный запрос!\n\n"
                "Следующий откроет платный доступ 👇"
            )
        elif left == 0:
            await message.answer("⚠️ Это было последнее сообщение.")

    await bot.send_chat_action(message.chat.id, "typing")

    try:
        answer = await ask_nvidia(user_id, message.text)
        answer = format_answer(answer)
        
        # Добавляем микро-продажу в конец
        if tier != "pro":
            answer += "\n\n💡 Хочешь ещё? Напиши следующий запрос"
        
        # Разбиваем длинные сообщения
        parts = split_text(answer)
        for part in parts:
            await message.answer(part)
    except Exception as e:
        logger.error(f"Ошибка NVIDIA API: {e}")
        await message.answer(
            "⚠️ Сервер загружен, попробуй ещё раз через пару секунд\n\n"
            "Если проблема повторяется — напиши /help"
        )


@dp.message()
async def handle_other(message: Message) -> None:
    await message.answer("Я понимаю только текстовые сообщения.")


# ─── Запуск ──────────────────────────────────────────────────────────────────
async def main() -> None:
    logger.info(f"Бот запущен. Модель: {NVIDIA_MODEL}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
