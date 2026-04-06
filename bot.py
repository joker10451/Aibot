import asyncio
import base64
import json
import logging
import os
import tempfile
from collections import defaultdict
from openai import OpenAI
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery,
    LabeledPrice, PreCheckoutQuery, ReplyKeyboardMarkup, KeyboardButton,
)
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from aiohttp import web
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
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")  # raw JSON or base64 JSON

if not TELEGRAM_TOKEN or not NVIDIA_API_KEY:
    raise ValueError("Заполни TELEGRAM_TOKEN и NVIDIA_API_KEY в файле .env")

# ─── Firebase Firestore ───────────────────────────────────────────────────────
def _resolve_firebase_credentials_path() -> str:
    """
    Поддержка Render/CI: можно передать credentials через env,
    чтобы не хранить ключи сервис-аккаунта как файл в репозитории.
    """
    if not FIREBASE_CREDENTIALS_JSON:
        return FIREBASE_CREDENTIALS

    raw = FIREBASE_CREDENTIALS_JSON.strip()
    try:
        if raw.startswith("{"):
            payload = json.loads(raw)
        else:
            decoded = base64.b64decode(raw).decode("utf-8")
            payload = json.loads(decoded)
    except Exception as e:
        raise ValueError(f"Некорректный FIREBASE_CREDENTIALS_JSON: {e}")

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "firebase_credentials.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return tmp_path


cred = credentials.Certificate(_resolve_firebase_credentials_path())
firebase_admin.initialize_app(cred)
db = firestore.client()
# Коллекции:
#   users/{user_id}  → { messages_used, paid, model, username }
#   history/{user_id}/messages/{doc_id} → { role, content, ts }
# ─────────────────────────────────────────────────────────────────────────────

# ─── Analytics events (минимум для воронки) ───────────────────────────────────
def _events_ref():
    return db.collection("events")


async def log_event(user_id: int | None, event: str, meta: dict | None = None) -> None:
    """Пишем событие в Firestore. Ошибки не должны ломать бота."""
    payload = {
        "event": event,
        "ts": firestore.SERVER_TIMESTAMP,
    }
    if user_id is not None:
        payload["user_id"] = int(user_id)
    if meta:
        payload["meta"] = meta

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: _events_ref().add(payload))
    except Exception as e:
        logger.warning(f"Не удалось записать event={event}: {e}")

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
HOMEWORK_BASE_PROMPT = (
    "Ты учебный ассистент. Решай задачи и объясняй так, чтобы понял ученик.\n\n"
    "Правила:\n"
    "— Без символов ** и любой markdown-разметки\n"
    "— Используй эмодзи для структуры и короткие абзацы\n"
    "— Если данных не хватает — задай 1–3 уточняющих вопроса\n"
    "— Не выдумывай факты, формулы и условия\n\n"
    "Качество:\n"
    "— Если не уверен, прямо скажи, что нужна проверка\n"
    "— Для математики всегда показывай промежуточные шаги\n"
    "— Избегай длинной воды, пиши кратко и по делу\n\n"
    "Формат ответа (если уместно):\n"
    "✅ Ответ: ...\n"
    "🧩 Решение по шагам:\n"
    "1) ...\n"
    "2) ...\n"
    "🧠 Объяснение простыми словами: ...\n"
    "🔎 Проверка / типичные ошибки: ...\n"
    "📝 Краткий конспект: ...\n"
)

MODES = {
    "📚 Домашка": HOMEWORK_BASE_PROMPT,
    "✍️ Тексты": "Ты профессиональный копирайтер, пишешь тексты. Используй эмодзи для структуры. Без символов ** и markdown.",
}

HOMEWORK_ACTIONS: dict[str, str] = {
    "🧩 Решение пошагово": "Сделай решение максимально пошаговым, без пропусков, с пояснениями каждого шага.",
    "🧠 Объясни проще": "Объясни очень простыми словами, как для новичка, с аналогиями и примерами.",
    "🔎 Проверь ответ": "Проверь решение/ответ пользователя: найди ошибки, объясни где и как исправить, потом дай правильный вариант.",
    "📝 Краткий конспект": "В конце добавь очень короткий конспект по теме (5–7 пунктов).",
}

user_modes = {}  # {user_id: system_prompt}
user_last_request = {}  # {user_id: timestamp} — защита от спама
user_pending_homework_action = {}  # {user_id: action_instruction}
user_last_prompt = {}  # {user_id: last_user_text} для /regen

PAYWALL_VARIANTS = {
    "A": (
        "🔒 Лимит бесплатных сообщений закончился.\n\n"
        f"🚀 Продолжай без остановки:\n"
        f"• Basic — {TIERS['basic']['price']} ⭐: 100 сообщений/30 дней\n"
        f"• Pro — {TIERS['pro']['price']} ⭐: безлимит + все модели\n\n"
        "Нажми кнопку и продолжай прямо сейчас 👇"
    ),
    "B": (
        "⛔ Бесплатный лимит исчерпан.\n\n"
        "Что дальше:\n"
        f"• Basic ({TIERS['basic']['price']} ⭐) — хватит на ежедневную учёбу\n"
        f"• Pro ({TIERS['pro']['price']} ⭐) — безлимит и максимум скорости\n\n"
        "Открой доступ за 1 минуту 👇"
    ),
}

BUY_SCREEN_VARIANTS = {
    "A": (
        f"💸 Разблокируй доступ за {STARS_PRICE} ⭐\n\n"
        "Что получишь сразу:\n"
        "— больше сообщений без пауз\n"
        "— доступ к более мощным моделям\n"
        "— быстрые ответы для учёбы\n\n"
        "Нажми кнопку ниже:"
    ),
    "B": (
        f"🚀 Открой доступ за {STARS_PRICE} ⭐\n\n"
        "Это удобно, если ты учишься каждый день:\n"
        "— решения по шагам\n"
        "— проверка ответов\n"
        "— короткие конспекты\n\n"
        "Жми кнопку ниже:"
    ),
}

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
        "mode": "📚 Домашка",
        "last_reset": datetime.datetime.now(datetime.timezone.utc),
    }
    await loop.run_in_executor(None, lambda: _user_ref(user_id).set(data))
    return data


async def set_user_mode(user_id: int, mode_key: str) -> None:
    """Сохраняет режим в памяти и Firestore."""
    if mode_key not in MODES:
        return
    user_modes[user_id] = MODES[mode_key]
    await update_user(user_id, {"mode": mode_key})


async def update_user(user_id: int, data: dict) -> None:
    """Обновить поля пользователя в Firestore."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: _user_ref(user_id).update(data))


async def check_access_and_maybe_increment(user_id: int) -> dict:
    """
    Атомарно проверяет лимит и инкрементит `messages_used` (кроме pro).
    Также делает месячный reset для basic/pro в той же транзакции.

    Возвращает:
      {
        allowed: bool,
        tier: str,
        prev_used: int,
        new_used: int,
        left_after: int,
        did_reset: bool,
      }
    """
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    user_ref = _user_ref(user_id)

    def _tx():
        transaction = db.transaction()

        @firestore.transactional
        def _run(transaction_obj):
            snap = user_ref.get(transaction=transaction_obj)
            data = snap.to_dict() or {}

            tier = data.get("tier", "free")
            limit = TIERS.get(tier, TIERS["free"])["limit"]
            prev_used = int(data.get("messages_used", 0) or 0)

            did_reset = False
            if tier in {"basic", "pro"}:
                last_reset = data.get("last_reset")
                if last_reset and (now - last_reset).days >= 30:
                    prev_used = 0
                    did_reset = True
                    transaction_obj.update(user_ref, {"messages_used": 0, "last_reset": now})

            # Pro — безлимит, не считаем сообщения
            if limit == -1:
                return {
                    "allowed": True,
                    "tier": tier,
                    "prev_used": prev_used,
                    "new_used": prev_used,
                    "left_after": 999,
                    "did_reset": did_reset,
                }

            # Если лимит исчерпан — не инкрементим
            if prev_used >= limit:
                return {
                    "allowed": False,
                    "tier": tier,
                    "prev_used": prev_used,
                    "new_used": prev_used,
                    "left_after": 0,
                    "did_reset": did_reset,
                }

            new_used = prev_used + 1
            transaction_obj.update(user_ref, {"messages_used": new_used})
            left_after = max(0, limit - new_used)

            return {
                "allowed": True,
                "tier": tier,
                "prev_used": prev_used,
                "new_used": new_used,
                "left_after": left_after,
                "did_reset": did_reset,
            }

        return _run(transaction)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _tx)

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
    # Убираем markdown-символы, которые часто проскакивают из моделей
    text = (
        text.replace("**", "")
        .replace("__", "")
        .replace("`", "")
    )

    # Убираем markdown-заголовки и bullets в начале строк
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            while stripped.startswith("#"):
                stripped = stripped[1:].lstrip()
        if stripped.startswith("* "):
            stripped = "• " + stripped[2:].lstrip()
        elif stripped.startswith("- "):
            stripped = "• " + stripped[2:].lstrip()
        elif stripped == "*":
            stripped = ""
        cleaned_lines.append(stripped if line == stripped else (line[: len(line) - len(line.lstrip())] + stripped))
    text = "\n".join(cleaned_lines)
    
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
    
    # Финальная подчистка одиночных звездочек внутри текста
    text = text.replace("*", "")
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


def pick_paywall_variant(user_id: int) -> str:
    """Стабильно выбирает вариант paywall для пользователя (A/B)."""
    return "A" if (user_id % 2 == 0) else "B"

def pick_buy_variant(user_id: int) -> str:
    """Стабильно выбирает вариант buy-screen для пользователя (A/B)."""
    # Можно оставить тот же сплит, чтобы не путать сегментацию
    return "A" if (user_id % 2 == 0) else "B"


async def ask_nvidia(user_id: int, user_text: str, append_user_message: bool = True) -> str:
    """Запрос к NVIDIA NIM API с историей из Firestore."""
    # Сохраняем сообщение пользователя (для обычных запросов)
    if append_user_message:
        await append_history(user_id, "user", user_text)

    # Загружаем историю для контекста
    history = await get_history(user_id)

    # Получаем модель пользователя
    user_data = await get_user(user_id)
    model = user_data.get("model", NVIDIA_MODEL)

    # Восстанавливаем режим из Firestore (если кэш пуст)
    if user_id not in user_modes:
        mode_key = user_data.get("mode") or "📚 Домашка"
        if mode_key in MODES:
            user_modes[user_id] = MODES[mode_key]

    # Получаем system prompt (режим + разовая учебная инструкция)
    base_prompt = user_modes.get(user_id)
    if not base_prompt:
        base_prompt = SYSTEM_PROMPT["content"]

    action = user_pending_homework_action.pop(user_id, None)
    if action:
        system_prompt = {
            "role": "system",
            "content": f"{base_prompt}\n\nДоп. требование:\n{action}",
        }
    else:
        system_prompt = {"role": "system", "content": base_prompt}

    loop = asyncio.get_event_loop()
    # Мягкий retry на временные сбои API
    last_error = None
    response = None
    for attempt in range(2):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: nvidia_client.chat.completions.create(
                    model=model,
                    messages=[system_prompt] + history,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                ),
            )
            break
        except Exception as e:
            last_error = e
            if attempt == 0:
                await asyncio.sleep(1.2)
            else:
                raise last_error

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
        [KeyboardButton(text="📚 Домашка"), KeyboardButton(text="✍️ Тексты")],
        [KeyboardButton(text="🧩 Решение пошагово"), KeyboardButton(text="🧠 Объясни проще")],
        [KeyboardButton(text="🔎 Проверь ответ"), KeyboardButton(text="📝 Краткий конспект")],
        [KeyboardButton(text="🔄 Перегенерировать")],
        [KeyboardButton(text="📊 Статус"), KeyboardButton(text="🧹 Очистить")],
    ])
    return kb


def get_quick_start_menu() -> ReplyKeyboardMarkup:
    """Быстрый старт с готовыми сценариями."""
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [KeyboardButton(text="✍️ Написать текст"), KeyboardButton(text="📚 Сделать домашку")],
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

    await log_event(user_id, "start")

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
        "📚 AskNeuro AI — помощник по учёбе\n\n"
        f"{access_line}\n"
        f"📦 Текущий тариф: {tier_name}\n\n"
        "Скинь задачу/тему — я:\n"
        "✅ решу и объясню по шагам\n"
        "🧠 объясню простыми словами\n"
        "🔎 проверю твой ответ и найду ошибки\n"
        "📝 сделаю короткий конспект\n\n"
        "⚡ Как пользоваться за 10 секунд:\n"
        "1) Нажми «📚 Сделать домашку» или просто напиши задачу\n"
        "2) При необходимости выбери «🧩 Пошагово» / «🧠 Объясни проще»\n"
        "3) Получи готовое решение и разбор\n\n"
        "Примеры:\n"
        "— «Реши: x² + 5x + 6 = 0»\n"
        "— «Объясни фотосинтез простыми словами»\n"
        "— «Проверь моё решение: ...»\n\n"
        "Выбери сценарий ниже или отправь задачу сразу:",
        reply_markup=get_quick_start_menu(),
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "🤖 AskNeuro AI — учебный помощник в Telegram.\n\n"
        "Что умеет:\n"
        "• решает задачи по шагам\n"
        "• объясняет темы простыми словами\n"
        "• проверяет ответы и находит ошибки\n"
        "• делает короткие конспекты\n\n"
        "📦 Тарифы:\n"
        "• Free: 5 сообщений, 2 базовые модели\n"
        "• Basic: 100 сообщений/месяц, 4 модели (100 ⭐)\n"
        "• Pro: безлимит, все 7 моделей (200 ⭐)\n\n"
        "Полезные команды:\n"
        "/start — быстрый старт\n"
        "/status — лимит и тариф\n"
        "/model — выбор модели\n"
        "/clear — очистить историю"
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
        await message.answer(f"✅ Тариф: Pro (безлимит)\nМодель: {model}")
    else:
        left = remaining_from(user_data)
        limit = TIERS[tier]["limit"]
        status_text = (
            f"📦 Тариф: {tier_name}\n"
            f"📊 Использовано: {user_data.get('messages_used', 0)}/{limit}\n"
            f"Осталось: {left}\nМодель: {model}"
        )
        
        if left == 0:
            status_text += (
                "\n\n🚀 Лимит исчерпан.\n"
                f"Открой Basic за {TIERS['basic']['price']} ⭐ (100 сообщений/30 дней) "
                f"или Pro за {TIERS['pro']['price']} ⭐ (безлимит)."
            )
        
        await message.answer(
            status_text,
            reply_markup=buy_keyboard() if left == 0 else None,
        )


@dp.message(Command("model"))
async def cmd_model(message: Message) -> None:
    user_data = await get_user(message.from_user.id)
    tier = user_data.get("tier", "free")
    await message.answer(
        f"🤖 Текущая модель: {user_data.get('model', NVIDIA_MODEL)}\n"
        f"📦 Тариф: {TIERS[tier]['name']}\n\nВыбери новую:",
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
    await log_event(call.from_user.id, "model_changed", {"model": model_id})
    available = get_available_models(user_data)
    label = next((k for k, v in available.items() if v == model_id), model_id)
    await call.message.answer(f"✅ Модель: {label}\n{model_id}")
    await call.answer()


@dp.callback_query(F.data == "buy")
async def callback_buy(call: CallbackQuery) -> None:
    """Отправляем инвойс Telegram Stars при нажатии кнопки."""
    await log_event(call.from_user.id, "invoice_sent", {"price": STARS_PRICE})
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
        "paid": tier != "free",
        "messages_used": 0,
        "last_reset": datetime.datetime.now(datetime.timezone.utc),
    })
    await log_event(user_id, "payment_success", {"stars": stars, "tier": tier})

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
    user_id = message.from_user.id
    variant = pick_buy_variant(user_id)
    await log_event(user_id, "buy_screen_shown", {"variant": variant})
    await message.answer(
        BUY_SCREEN_VARIANTS[variant],
        reply_markup=buy_keyboard(),
    )


@dp.message(Command("regen"))
async def cmd_regen(message: Message) -> None:
    """/regen — перегенерировать последний ответ."""
    user_id = message.from_user.id
    last_prompt = user_last_prompt.get(user_id)
    if not last_prompt:
        await message.answer("Сначала отправь запрос, потом можно сделать перегенерацию.")
        return
    await handle_regen(message, user_id, last_prompt)


@dp.message(F.text == "🔄 Перегенерировать")
async def menu_regen(message: Message) -> None:
    """Кнопка перегенерации последнего ответа."""
    user_id = message.from_user.id
    last_prompt = user_last_prompt.get(user_id)
    if not last_prompt:
        await message.answer("Сначала отправь любой запрос.")
        return
    await handle_regen(message, user_id, last_prompt)


async def handle_regen(message: Message, user_id: int, last_prompt: str) -> None:
    """Общая логика перегенерации ответа."""
    await bot.send_chat_action(message.chat.id, "typing")
    regen_prompt = (
        f"Сделай альтернативный вариант ответа на этот запрос.\n\nЗапрос пользователя:\n{last_prompt}"
    )
    try:
        answer = await ask_nvidia(user_id, regen_prompt, append_user_message=False)
        answer = format_answer(answer)
        await message.answer(answer)
    except Exception as e:
        logger.error(f"Ошибка regenerate: {e}")
        await log_event(user_id, "error_nvidia", {"error": str(e)[:300], "source": "regen"})
        await message.answer("Не удалось перегенерировать сейчас. Попробуй ещё раз через пару секунд.")


# ─── Обработчики кнопок меню ─────────────────────────────────────────────────

@dp.message(F.text == "✍️ Написать текст")
async def quick_write_text(message: Message) -> None:
    """Быстрый старт: написать текст."""
    await set_user_mode(message.from_user.id, "✍️ Тексты")
    await log_event(message.from_user.id, "mode_selected", {"mode": "✍️ Тексты"})
    await message.answer(
        "✍️ Напиши тему — я создам текст\n\nНапример:\n— \"пост про путешествия\"\n— \"сочинение про экологию\"",
        reply_markup=get_main_menu(),
    )


@dp.message(F.text == "📚 Сделать домашку")
async def quick_homework(message: Message) -> None:
    """Быстрый старт: домашка."""
    await set_user_mode(message.from_user.id, "📚 Домашка")
    await log_event(message.from_user.id, "mode_selected", {"mode": "📚 Домашка"})
    await message.answer(
        "📚 Напиши задачу — я решу и объясню\n\nНапример:\n— \"реши уравнение x² + 5x + 6 = 0\"\n— \"объясни фотосинтез\"",
        reply_markup=get_main_menu(),
    )


@dp.message(F.text.in_(list(MODES.keys())))
async def set_mode(message: Message) -> None:
    """Установка режима работы бота."""
    user_id = message.from_user.id
    mode_text = message.text
    await set_user_mode(user_id, mode_text)
    await log_event(user_id, "mode_selected", {"mode": mode_text})
    
    await message.answer(
        f"✅ Режим выбран: {mode_text}\n\nТеперь напиши свой запрос 👇",
        reply_markup=get_main_menu(),
    )


@dp.message(F.text.in_(list(HOMEWORK_ACTIONS.keys())))
async def set_homework_action(message: Message) -> None:
    """Быстрые учебные действия (пошагово/проще/проверка/конспект)."""
    user_id = message.from_user.id
    action = message.text
    await set_user_mode(user_id, "📚 Домашка")
    user_pending_homework_action[user_id] = HOMEWORK_ACTIONS[action]

    if action == "🔎 Проверь ответ":
        hint = (
            "🔎 Ок! Пришли своё решение/ответ целиком.\n"
            "Если есть условие задачи — добавь его сверху."
        )
    else:
        hint = (
            "📚 Ок! Пришли задачу/тему.\n"
            "Чем точнее условие, тем лучше результат."
        )

    await message.answer(hint, reply_markup=get_main_menu())


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
    """Выдать доступ: /grant <user_id> [free|basic|pro]"""
    if message.from_user.id != ADMIN_ID:
        return
    args = message.text.split()
    if len(args) < 2:
        await message.answer("Использование: /grant <user_id> [free|basic|pro]")
        return
    try:
        target_id = int(args[1])
        tier = (args[2].strip().lower() if len(args) >= 3 else "pro")
        if tier not in TIERS:
            await message.answer("❌ Неверный tier. Используй: free, basic или pro")
            return

        import datetime
        await update_user(target_id, {
            "tier": tier,
            "paid": tier != "free",
            "messages_used": 0,
            "last_reset": datetime.datetime.now(datetime.timezone.utc),
        })
        await message.answer(f"✅ Тариф установлен: {target_id} → {TIERS[tier]['name']}")
        if tier == "pro":
            note = "🎉 Доступ активирован! Теперь у тебя Pro (безлимит)."
        elif tier == "basic":
            note = "🎉 Доступ активирован! Теперь у тебя Basic (100 сообщений/30 дней)."
        else:
            note = "ℹ️ Тариф изменён на Free."
        await bot.send_message(target_id, note)
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
        paid    = sum(1 for u in users if u.get("tier") in {"basic", "pro"})
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


@dp.message(Command("funnel"))
async def cmd_funnel(message: Message) -> None:
    """/funnel [days] — воронка событий за N дней (по умолчанию 7)."""
    if message.from_user.id != ADMIN_ID:
        return

    args = message.text.split()
    days = 7
    if len(args) >= 2:
        try:
            days = int(args[1])
        except ValueError:
            days = 7
    days = max(1, min(days, 60))

    import datetime
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    event_names = [
        "start",
        "first_message",
        "paywall_shown",
        "invoice_sent",
        "payment_success",
    ]

    loop = asyncio.get_event_loop()

    def _load_recent_events() -> list[dict]:
        docs = (
            db.collection("events")
            .where("ts", ">=", since)
            .order_by("ts", direction=firestore.Query.DESCENDING)
            .limit(5000)
            .get()
        )
        return [d.to_dict() for d in docs]

    try:
        events = await loop.run_in_executor(None, _load_recent_events)
    except Exception as e:
        await message.answer(f"Ошибка чтения events: {e}")
        return

    counts = {k: 0 for k in event_names}
    for ev in events:
        name = ev.get("event")
        if name in counts:
            counts[name] += 1

    start = counts["start"]
    first = counts["first_message"]
    paywall = counts["paywall_shown"]
    invoice = counts["invoice_sent"]
    paid = counts["payment_success"]
    unique_users = set()
    paywall_variants = defaultdict(int)
    for ev in events:
        uid = ev.get("user_id")
        if uid is not None:
            unique_users.add(uid)
        if ev.get("event") == "paywall_shown":
            meta = ev.get("meta") or {}
            variant = meta.get("variant")
            if variant in {"A", "B"}:
                paywall_variants[variant] += 1

    def pct(num: int, den: int) -> str:
        if den <= 0:
            return "—"
        return f"{(num / den) * 100:.1f}%"

    text = (
        f"Воронка за {days} дн.\n\n"
        f"Уникальных пользователей: {len(unique_users)}\n"
        f"start: {start}\n"
        f"first_message: {first} (от start: {pct(first, start)})\n"
        f"paywall_shown: {paywall} (от first: {pct(paywall, first)})\n"
        f"paywall A/B: A={paywall_variants['A']} | B={paywall_variants['B']}\n"
        f"invoice_sent: {invoice} (от paywall: {pct(invoice, paywall)})\n"
        f"payment_success: {paid} (от invoice: {pct(paid, invoice)})\n\n"
        f"Итог start→payment: {pct(paid, start)}\n"
        "Примечание: считаются события из `events` (ограничение выборки 5000 записей)."
    )
    await message.answer(text)


@dp.message(Command("revenue"))
async def cmd_revenue(message: Message) -> None:
    """/revenue [days] — агрегат по оплатам и Stars."""
    if message.from_user.id != ADMIN_ID:
        return
    args = message.text.split()
    days = 7
    if len(args) >= 2:
        try:
            days = int(args[1])
        except ValueError:
            days = 7
    days = max(1, min(days, 90))

    import datetime
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    loop = asyncio.get_event_loop()

    def _load_payment_events() -> list[dict]:
        docs = (
            db.collection("events")
            .where("event", "==", "payment_success")
            .where("ts", ">=", since)
            .order_by("ts", direction=firestore.Query.DESCENDING)
            .limit(5000)
            .get()
        )
        return [d.to_dict() for d in docs]

    try:
        payments = await loop.run_in_executor(None, _load_payment_events)
    except Exception as e:
        await message.answer(f"Ошибка чтения оплат: {e}")
        return

    total_stars = 0
    total_payments = len(payments)
    by_tier = defaultdict(int)
    for ev in payments:
        meta = ev.get("meta") or {}
        stars = int(meta.get("stars", 0) or 0)
        tier = str(meta.get("tier", "unknown"))
        total_stars += stars
        by_tier[tier] += 1

    await message.answer(
        f"Выручка за {days} дн.\n\n"
        f"Оплат: {total_payments}\n"
        f"Stars: {total_stars} ⭐\n"
        f"По тарифам: basic={by_tier['basic']}, pro={by_tier['pro']}, other={by_tier['unknown']}"
    )


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

    # Атомарно проверяем лимит и увеличиваем счётчик
    state = await check_access_and_maybe_increment(user_id)
    tier = state.get("tier", user_data.get("tier", "free"))

    if not state.get("allowed", False):
        variant = pick_paywall_variant(user_id)
        await log_event(user_id, "paywall_shown", {"tier": tier, "variant": variant})
        await message.answer(
            PAYWALL_VARIANTS[variant],
            reply_markup=buy_keyboard(),
        )
        return

    # ВАУ-эффект для первого сообщения
    if int(state.get("prev_used", 0) or 0) == 0:
        await log_event(user_id, "first_message", {"tier": tier})
        await message.answer("⚡ Сейчас покажу, что я умею...")

    if tier != "pro":
        left_after = int(state.get("left_after", 0) or 0)
        if left_after == 1:
            await log_event(user_id, "free_limit_warning", {"tier": tier, "left": 1})
            await message.answer(
                f"⚠️ Остался 1 бесплатный запрос.\n\n"
                f"Дальше: Basic {TIERS['basic']['price']} ⭐ или Pro {TIERS['pro']['price']} ⭐."
            )
        elif left_after == 0:
            await message.answer("⚠️ Это был последний бесплатный ответ. Готов открыть доступ?")

    await bot.send_chat_action(message.chat.id, "typing")

    try:
        user_last_prompt[user_id] = message.text
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
        await log_event(user_id, "error_nvidia", {"error": str(e)[:300]})
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

    # Render Web Service ожидает открытый порт. Поднимаем минимальный health endpoint.
    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", "10000"))
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    logger.info(f"Health endpoint: /health on port {port}")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
