import asyncio
import base64
import json
import logging
import os
import tempfile
from collections import defaultdict, deque
import re
import io
from typing import Optional
from pypdf import PdfReader
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
NVIDIA_MODEL         = os.getenv("NVIDIA_MODEL", "mistralai/mistral-7b-instruct-v0.3")
# Принудительно используем рабочую модель для стабильности
NVIDIA_MODEL = "z-ai/glm4_7"
ADMIN_USERNAME       = os.getenv("ADMIN_USERNAME", "@your_username")
ADMIN_ID             = int(os.getenv("ADMIN_ID", 0))  # твой Telegram user_id
STARS_PRICE          = int(os.getenv("STARS_PRICE", 100))  # цена в Stars
FREE_LIMIT           = int(os.getenv("FREE_LIMIT", 5))
MAX_HISTORY          = int(os.getenv("MAX_HISTORY", 20))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", 800))  # Снижено для экономии
TEMPERATURE          = float(os.getenv("TEMPERATURE", 0.7))
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS", "firebase.json")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")  # raw JSON or base64 JSON
OCR_SPACE_API_KEY    = os.getenv("OCR_SPACE_API_KEY")  # OCR для картинок через ocr.space (опционально)

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


def _feedback_ref():
    return db.collection("feedback")


async def save_feedback(user_id: int, payload: dict) -> None:
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: _feedback_ref().add(payload))
    except Exception as e:
        logger.warning(f"Не удалось записать feedback: {e}")


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
    "— Избегай длинной воды, пиши кратко и по делу\n"
    "— Одна мысль = один короткий абзац (1–3 строки)\n"
    "— По умолчанию отвечай КОРОТКО: без длинных эссе\n"
    "— Длинные блоки (типичные ошибки/конспект/очень подробное объяснение) давай только если пользователь попросил\n"
    "— Не пиши обесценивающие фразы вроде «все ошибаются» или «неправильно решать через формулу»\n\n"
    "Формат ответа по умолчанию:\n"
    "✅ Ответ: ... (1 строка)\n"
    "🧩 Шаги: 2–5 коротких пунктов\n"
    "📝 Итог: 1–2 строки как оформить/что запомнить\n\n"
    "Если пользователь попросил конкретный формат (пошагово/проще/проверить/конспект) — следуй запросу."
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
    "📄 Оформи как для сдачи": "Оформи решение так, чтобы его можно было сдать: аккуратная структура, обозначения, проверка, итоговый ответ. Коротко и без воды.",
    "✅ Проверь оформление": "Проверь оформление и логику: где не хватает пояснений/шагов/обозначений. Дай список улучшений и затем покажи как правильно оформить.",
}

user_modes = {}  # {user_id: system_prompt}
user_last_request = {}  # {user_id: timestamp} — защита от спама
user_pending_homework_action = {}  # {user_id: action_instruction}
user_last_prompt = {}  # {user_id: last_user_text} для /regen
user_recent_answers = defaultdict(lambda: deque(maxlen=30))  # {user_id: deque[timestamps]} для мини-статов перед paywall
user_pending_ocr = {}  # {user_id: {"text": str, "conf": float | None, "ts": float}}
user_last_exchange = {}  # {user_id: {"prompt": str, "answer": str, "ts": float, "mode": str, "model": str}}

PAYWALL_VARIANTS = {
    "A": (
        "🔒 Лимит бесплатных сообщений закончился.\n\n"
        "Что откроется после оплаты:\n"
        "• 📸 задачи по фото\n"
        "• ✅ проверка решений и оформление «как для сдачи»\n"
        "• 🧩 подробности по кнопкам (пошагово/проще/ошибки/конспект)\n\n"
        f"🚀 Тарифы:\n"
        f"• Basic — {TIERS['basic']['price']} ⭐: 100 сообщений/30 дней\n"
        f"• Pro — {TIERS['pro']['price']} ⭐: безлимит + все модели\n\n"
        "💡 Нужны Stars? Быстрая покупка:\n"
        "👉 https://t.me/onelinkgo_bot?start=_tgr_KBB2ywM0Mjky\n\n"
        "Нажми кнопку и продолжай 👇"
    ),
    "B": (
        "⛔ Бесплатный лимит исчерпан.\n\n"
        "Продолжить учёбу без ограничений:\n"
        "• оформление решений «как у преподавателя»\n"
        "• проверка ошибок и типичных ловушек\n"
        "• задачи с фото (сканы/тетрадь)\n\n"
        f"Выбери тариф:\n"
        f"• Basic ({TIERS['basic']['price']} ⭐) — 100 сообщений/30 дней\n"
        f"• Pro ({TIERS['pro']['price']} ⭐) — безлимит + все модели\n\n"
        "⭐ Нет Stars? Купи через партнера:\n"
        "👉 https://t.me/onelinkgo_bot?start=_tgr_KBB2ywM0Mjky\n\n"
        "Открой доступ 👇"
    ),
}

BUY_SCREEN_VARIANTS = {
    "A": (
        f"💸 Разблокируй доступ за {STARS_PRICE} ⭐\n\n"
        "Что получишь сразу:\n"
        "— больше сообщений без пауз\n"
        "— доступ к более мощным моделям\n"
        "— быстрые ответы для учёбы\n\n"
        "💡 Нужны Stars? Быстрая покупка через бота:\n"
        "👉 https://t.me/onelinkgo_bot?start=_tgr_KBB2ywM0Mjky\n\n"
        "Нажми кнопку ниже:"
    ),
    "B": (
        f"🚀 Открой доступ за {STARS_PRICE} ⭐\n\n"
        "Это удобно, если ты учишься каждый день:\n"
        "— решения по шагам\n"
        "— проверка ответов\n"
        "— короткие конспекты\n\n"
        "⭐ Нет Stars? Купи через нашего партнера:\n"
        "👉 https://t.me/onelinkgo_bot?start=_tgr_KBB2ywM0Mjky\n\n"
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
        "model_auto": True,
        "bonus_messages": 0,
        "referred_by": None,
        "referral_rewarded": False,
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
            bonus = int(data.get("bonus_messages", 0) or 0)

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

            effective_limit = limit + max(0, bonus)

            # Если лимит исчерпан — не инкрементим
            if prev_used >= effective_limit:
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
            left_after = max(0, effective_limit - new_used)

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


# ─── OCR / извлечение текста ─────────────────────────────────────────────────

def _is_probably_low_quality_ocr(text: str, conf: Optional[float]) -> bool:
    t = (text or "").strip()
    if len(t) < 25:
        return True
    if conf is not None and conf < 0.6:
        return True
    letters = sum(ch.isalnum() for ch in t)
    return letters / max(1, len(t)) < 0.35


async def ocr_space_image(image_bytes: bytes, language: str = "rus") -> tuple[str, Optional[float]]:
    """
    OCR через ocr.space. Требует OCR_SPACE_API_KEY.
    Возвращает (text, mean_confidence[0..1] | None)
    """
    if not OCR_SPACE_API_KEY:
        raise RuntimeError("OCR_SPACE_API_KEY не задан")

    import aiohttp

    url = "https://api.ocr.space/parse/image"
    data = aiohttp.FormData()
    data.add_field("apikey", OCR_SPACE_API_KEY)
    data.add_field("language", language)
    data.add_field("isOverlayRequired", "true")
    data.add_field("file", image_bytes, filename="image.jpg", content_type="image/jpeg")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15, connect=10)) as session:
        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                raise RuntimeError(f"OCR API HTTP {resp.status}")
            payload = await resp.json(content_type=None)

    # Логируем ответ для диагностики
    if isinstance(payload, dict) and payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "OCR error"
        logger.warning(f"OCR error: {msg}")
        raise RuntimeError(str(msg))

    # Проверяем структуру ответа
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid OCR response: {type(payload)}")

    parsed = payload.get("ParsedResults") or []
    if not parsed:
        # Проверяем есть ли ошибка в OCRExitCode
        exit_code = payload.get("OCRExitCode")
        if exit_code != 1:
            logger.warning(f"OCR exit code: {exit_code}, response: {str(payload)[:200]}")
        return "", None

    text = "\n".join((p.get("ParsedText") or "") for p in parsed).strip()

    conf_vals: list[float] = []
    try:
        for p in parsed:
            overlay = p.get("TextOverlay") or {}
            for ln in (overlay.get("Lines") or []):
                for w in (ln.get("Words") or []):
                    c = w.get("WordConf")
                    if c is None:
                        continue
                    conf_vals.append(float(c) / 100.0)
    except Exception:
        conf_vals = []

    conf = (sum(conf_vals) / len(conf_vals)) if conf_vals else None
    return text, conf


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: list[str] = []
    for page in reader.pages[:20]:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(out).strip()


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

    # Нормализуем переносы и визуальные блоки, чтобы не было "полотна"
    # 1) убираем лишние пробелы и >2 пустых строк подряд
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 2) Добавляем пустую строку перед "заголовками" с эмодзи
    #    и разрезаем случаи, когда модель склеила заголовки в одну строку
    heading_emojis = "✅🧩🧠🔎📝📌📆📊🚀❓💡🎯"
    text = re.sub(rf"(?<!\n)\s*([{heading_emojis}])\s*", r"\n\n\1 ", text)

    lines = text.splitlines()
    pretty = []
    for i, line in enumerate(lines):
        s = line.strip()
        is_heading = bool(re.match(r"^[✅🧩🧠🔎📝📌📆📊🚀❓💡🎯] ", s)) or bool(re.match(r"^[✅🧩🧠🔎📝📌📆📊🚀❓💡🎯]", s))
        if is_heading and pretty and pretty[-1].strip() != "":
            pretty.append("")
        pretty.append(line.rstrip())
    text = "\n".join(pretty)

    # 3) Чуть улучшаем читабельность списков
    #    - "1." и "1)" -> "1)"; лишние пробелы после маркеров
    text = re.sub(r"^(\s*\d+)\.\s+", r"\1) ", text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*•)\s+", "• ", text, flags=re.MULTILINE)

    # 4) Если текст всё равно выглядит монолитом, добавим мягкую разбивку после предложений в больших абзацах
    #    (только если в строке > 220 символов и нет явных переносов)
    wrapped_lines = []
    for line in text.splitlines():
        if len(line) > 220 and "•" not in line and not re.search(r"\d\)", line):
            # попробуем вставить перенос после ближайшей точки/двоеточия
            parts = re.split(r"(?<=[\.:;])\s+", line)
            buf = ""
            for part in parts:
                if not buf:
                    buf = part
                elif len(buf) + 1 + len(part) <= 160:
                    buf += " " + part
                else:
                    wrapped_lines.append(buf)
                    buf = part
            if buf:
                wrapped_lines.append(buf)
        else:
            wrapped_lines.append(line)
    text = "\n".join(wrapped_lines)

    # финальная нормализация
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


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
    bonus = int(user_data.get("bonus_messages", 0) or 0)
    effective_limit = limit + max(0, bonus)
    return max(0, effective_limit - user_data.get("messages_used", 0))


def get_available_models(user_data: dict) -> dict[str, str]:
    """Возвращает доступные модели для тарифа пользователя."""
    tier = user_data.get("tier", "free")
    return TIERS[tier]["models"]


def pick_model_for_request(user_data: dict, user_text: str) -> str:
    """
    Простая авто-маршрутизация модели.
    Если пользователь выбирал модель вручную (model_auto=False) — используем её.
    Иначе выбираем из доступных по тарифу.
    """
    available = list(get_available_models(user_data).values())
    if not available:
        return user_data.get("model", NVIDIA_MODEL) or NVIDIA_MODEL

    # Если пользователь вручную выбрал модель — не трогаем
    if user_data.get("model_auto") is False and user_data.get("model"):
        return user_data["model"]

    t = (user_text or "").lower()
    is_math = any(ch in user_text for ch in ["=", "√", "^", "∫", "Σ", "Δ", "π"]) or ("x^" in t) or ("x²" in t)
    is_code = any(k in t for k in ["python", "java", "c++", "javascript", "ошибка", "traceback", "stacktrace", "sql", "код"])
    long = len(user_text) > 600

    # приоритетные цели
    preferred = []
    if is_math or is_code or long:
        preferred = [
            "meta/llama-3.1-70b-instruct",
            "mistralai/mistral-7b-instruct-v0.3",
            "z-ai/glm4_7",
        ]
    else:
        preferred = [
            "mistralai/mistral-7b-instruct-v0.3",
            "meta/llama-3.1-70b-instruct",
            "z-ai/glm4_7",
        ]

    for mid in preferred:
        if mid in available:
            return mid

    return available[0]


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
    model = pick_model_for_request(user_data, user_text)
    if user_data.get("model") != model:
        # не блокируем ответ, если запись не прошла
        try:
            await update_user(user_id, {"model": model})
        except Exception:
            pass

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
            # Частый кейс: конкретная модель недоступна на endpoint (404).
            # Перебираем рабочие fallback-модели из доступных на тарифе.
            if "404" in str(e):
                available = list(get_available_models(user_data).values())
                fallback_candidates = [
                    "meta/llama-3.1-70b-instruct",
                    "mistralai/mistral-7b-instruct-v0.3",
                    "z-ai/glm4_7",
                ]
                switched = False
                for candidate in fallback_candidates:
                    if candidate in available and candidate != model:
                        model = candidate
                        switched = True
                        try:
                            await update_user(user_id, {"model": model, "model_auto": True})
                        except Exception:
                            pass
                        break
                if switched:
                    continue
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
        [
            InlineKeyboardButton(
                text=f"⭐ Basic ({TIERS['basic']['price']} Stars)",
                callback_data="buy:basic",
            ),
            InlineKeyboardButton(
                text=f"🚀 Pro ({TIERS['pro']['price']} Stars)",
                callback_data="buy:pro",
            ),
        ],
    ])


def homework_details_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🧩 Пошагово", callback_data="detail:steps"),
            InlineKeyboardButton(text="🧠 Проще", callback_data="detail:simple"),
        ],
        [
            InlineKeyboardButton(text="🔎 Ошибки", callback_data="detail:check"),
            InlineKeyboardButton(text="📝 Конспект", callback_data="detail:summary"),
        ],
    ])


def ocr_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Использовать текст", callback_data="ocr:confirm"),
            InlineKeyboardButton(text="✏️ Отмена", callback_data="ocr:cancel"),
        ]
    ])


def feedback_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⭐ 1", callback_data="fb:1"),
            InlineKeyboardButton(text="⭐ 2", callback_data="fb:2"),
            InlineKeyboardButton(text="⭐ 3", callback_data="fb:3"),
            InlineKeyboardButton(text="⭐ 4", callback_data="fb:4"),
            InlineKeyboardButton(text="⭐ 5", callback_data="fb:5"),
        ],
        [
            InlineKeyboardButton(text="⚠️ Плохо ответил", callback_data="fb:bad"),
        ],
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
        [KeyboardButton(text="📄 Оформи как для сдачи"), KeyboardButton(text="✅ Проверь оформление")],
        [KeyboardButton(text="🔄 Перегенерировать")],
        [KeyboardButton(text="📊 Статус"), KeyboardButton(text="🧹 Очистить")],
        [KeyboardButton(text="📤 Поделиться ботом")],
    ])
    return kb


def get_quick_start_menu() -> ReplyKeyboardMarkup:
    """Быстрый старт с готовыми сценариями."""
    kb = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [KeyboardButton(text="✍️ Написать текст"), KeyboardButton(text="📚 Сделать домашку")],
        [KeyboardButton(text="📊 Статус"), KeyboardButton(text="🧹 Очистить")],
        [KeyboardButton(text="📤 Поделиться ботом")],
    ])
    return kb


# ─── Бот и диспетчер ─────────────────────────────────────────────────────────
bot = Bot(token=TELEGRAM_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    user_id   = message.from_user.id
    # парсим referral payload: /start ref_<referrer_id>
    payload = ""
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) == 2:
        payload = parts[1].strip()

    referrer_id = None
    if payload.startswith("ref_"):
        raw = payload[4:].strip()
        if raw.isdigit():
            referrer_id = int(raw)

    user_data = await get_user(user_id)
    user_data = await check_and_reset_limits(user_id, user_data)

    await log_event(user_id, "start")

    # Сохраняем username при первом старте
    if not user_data.get("username"):
        await update_user(user_id, {"username": message.from_user.username or ""})

    # Сохраняем, кто пригласил (только 1 раз, без self-ref)
    if referrer_id and referrer_id != user_id and not user_data.get("referred_by"):
        await update_user(user_id, {"referred_by": referrer_id})
        await log_event(user_id, "referral_link_opened", {"referrer_id": referrer_id})
        user_data["referred_by"] = referrer_id

    tier = user_data.get("tier", "free")
    tier_name = TIERS[tier]["name"]
    left = remaining_from(user_data)

    if tier == "pro":
        access_line = "✅ У тебя Pro — безлимит"
    else:
        limit = TIERS[tier]["limit"]
        access_line = f"🆓 Бесплатно: {left} сообщений"

    await message.answer(
        "🤖 Бот, который помогает с домашкой\n\n"
        f"{access_line}\n"
        f"📦 Текущий тариф: {tier_name}\n\n"
        "Скинь задачу — получишь:\n"
        "✅ готовое решение\n"
        "🧠 объяснение простыми словами\n"
        "🔎 проверку ошибок (если пришлёшь свой ответ)\n\n"
        "⚡ Попробуй прямо сейчас:\n"
        "«Реши x² + 5x + 6 = 0»\n\n"
        "Выбери сценарий ниже или отправь задачу сразу 👇",
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
        "/clear — очистить историю\n"
        "/ref — реферальная ссылка"
    )


@dp.message(Command("ref"))
async def cmd_ref(message: Message) -> None:
    """Реферальная ссылка: пригласи друга и получи +5 сообщений."""
    me = await bot.get_me()
    user_id = message.from_user.id
    link = f"https://t.me/{me.username}?start=ref_{user_id}"
    await message.answer(
        "🎁 Рефералка: приведи друга — получишь +5 сообщений.\n\n"
        f"Твоя ссылка:\n{link}\n\n"
        "Условие: друг должен перейти по ссылке и отправить хотя бы 1 запрос.",
        reply_markup=get_main_menu(),
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
                f"или Pro за {TIERS['pro']['price']} ⭐ (безлимит).\n\n"
                "💡 Нужны Stars? Быстрая покупка:\n"
                "👉 https://t.me/onelinkgo_bot?start=_tgr_KBB2ywM0Mjky"
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


@dp.message(Command("auto"))
async def cmd_auto_model(message: Message) -> None:
    """Включить авто-выбор модели по задаче."""
    await update_user(message.from_user.id, {"model_auto": True})
    await message.answer("✅ Автомодель включена. Я буду выбирать модель под задачу.")


@dp.callback_query(F.data.startswith("model:"))
async def callback_model(call: CallbackQuery) -> None:
    model_id = call.data.split("model:", 1)[1]
    user_data = await get_user(call.from_user.id)
    
    # Проверка доступа к модели
    if not check_model_access(user_data, model_id):
        tier = user_data.get("tier", "free")
        await call.answer(f"❌ Эта модель недоступна на тарифе {TIERS[tier]['name']}", show_alert=True)
        return
    
    await update_user(call.from_user.id, {"model": model_id, "model_auto": False})
    await log_event(call.from_user.id, "model_changed", {"model": model_id})
    available = get_available_models(user_data)
    label = next((k for k, v in available.items() if v == model_id), model_id)
    await call.message.answer(f"✅ Модель: {label}\n{model_id}")
    await call.answer()


@dp.callback_query(F.data.startswith("buy"))
async def callback_buy(call: CallbackQuery) -> None:
    """Отправляем инвойс Telegram Stars при нажатии кнопки."""
    parts = (call.data or "").split(":", 1)
    tier = parts[1] if len(parts) == 2 else "basic"
    if tier not in ("basic", "pro"):
        tier = "basic"

    price = int(TIERS[tier]["price"])
    tier_name = TIERS[tier]["name"]

    await log_event(call.from_user.id, "invoice_sent", {"price": price, "tier": tier})
    await bot.send_invoice(
        chat_id=call.from_user.id,
        title="Доступ к AskNeuro AI",
        description=(
            f"Тариф {tier_name} — {price} ⭐\n"
            f"{'Безлимит + все модели' if tier == 'pro' else '100 сообщений/30 дней'}"
        ),
        payload=f"ai_access_stars:{tier}",
        provider_token="",        # пусто — это Telegram Stars (XTR)
        currency="XTR",
        prices=[LabeledPrice(label=f"{tier_name} доступ", amount=price)],
    )
    await call.answer()


@dp.callback_query(F.data.startswith("detail:"))
async def callback_homework_detail(call: CallbackQuery) -> None:
    user_id = call.from_user.id
    user_data = await get_user(user_id)

    # считаем как сообщение (потому что это генерация)
    state = await check_access_and_maybe_increment(user_id)
    tier = state.get("tier", user_data.get("tier", "free"))
    if not state.get("allowed", False):
        variant = pick_paywall_variant(user_id)
        await log_event(user_id, "paywall_shown", {"tier": tier, "variant": variant})
        await call.message.answer(PAYWALL_VARIANTS[variant], reply_markup=buy_keyboard())
        await call.answer()
        return

    last_prompt = user_last_prompt.get(user_id)
    if not last_prompt:
        await call.answer("Сначала отправь задачу текстом.", show_alert=True)
        return

    kind = (call.data or "").split("detail:", 1)[1].strip()
    action_map = {
        "steps": "Сделай решение максимально пошаговым, без пропусков. Покажи промежуточные шаги. Без воды.",
        "simple": "Объясни очень простыми словами, как для новичка, с аналогией/примером. Коротко.",
        "check": "Проверь решение и типичные ошибки для этой задачи. Короткий список ошибок и проверка ответа.",
        "summary": "Сделай очень короткий конспект по теме: 5–7 пунктов.",
    }
    action = action_map.get(kind)
    if not action:
        await call.answer()
        return

    await call.answer("Делаю…")
    await bot.send_chat_action(call.message.chat.id, "typing")

    regen_prompt = f"{last_prompt}\n\nДоп. требование:\n{action}"
    answer = await ask_nvidia(user_id, regen_prompt, append_user_message=False)
    answer = format_answer(answer)
    for part in split_text(answer):
        await call.message.answer(part)


@dp.callback_query(F.data == "ocr:cancel")
async def callback_ocr_cancel(call: CallbackQuery) -> None:
    user_pending_ocr.pop(call.from_user.id, None)
    await call.answer("Ок", show_alert=False)


@dp.callback_query(F.data == "ocr:confirm")
async def callback_ocr_confirm(call: CallbackQuery) -> None:
    user_id = call.from_user.id
    pending = user_pending_ocr.pop(user_id, None)
    if not pending or not (pending.get("text") or "").strip():
        await call.answer("Нечего подтверждать.", show_alert=True)
        return

    # Тут не инкрементим лимит: он уже списан при загрузке фото/файла
    text = pending["text"].strip()
    user_last_prompt[user_id] = text
    await call.answer("Делаю…")
    await bot.send_chat_action(call.message.chat.id, "typing")

    answer = await ask_nvidia(user_id, text)
    answer = format_answer(answer)
    for part in split_text(answer):
        await call.message.answer(part)


@dp.callback_query(F.data.startswith("fb:"))
async def callback_feedback(call: CallbackQuery) -> None:
    user_id = call.from_user.id
    data = (call.data or "").split("fb:", 1)[1].strip()
    exchange = user_last_exchange.get(user_id)
    if not exchange:
        await call.answer("Нет последнего ответа для оценки.", show_alert=True)
        return

    import time
    base = {
        "user_id": int(user_id),
        "ts": firestore.SERVER_TIMESTAMP,
        "client_ts": time.time(),
        "prompt": exchange.get("prompt", "")[:4000],
        "answer": exchange.get("answer", "")[:8000],
        "mode": exchange.get("mode", ""),
        "model": exchange.get("model", ""),
        "username": call.from_user.username or "",
    }

    if data == "bad":
        base["rating"] = 0
        base["label"] = "bad"
        await save_feedback(user_id, base)
        await log_event(user_id, "feedback_bad", {"mode": base["mode"], "model": base["model"]})
        await call.answer("Принял. Спасибо — улучшим.", show_alert=True)
        return

    if data.isdigit():
        rating = int(data)
        if 1 <= rating <= 5:
            base["rating"] = rating
            base["label"] = "stars"
            await save_feedback(user_id, base)
            await log_event(user_id, "feedback_star", {"rating": rating, "mode": base["mode"], "model": base["model"]})
            await call.answer("Спасибо за оценку!", show_alert=False)
            return

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


@dp.message(F.text == "📤 Поделиться ботом")
async def menu_share(message: Message) -> None:
    """Кнопка Поделиться ботом из меню."""
    me = await bot.get_me()
    user_id = message.from_user.id
    await message.answer(
        "🚀 Вот бот, который помогает с домашкой:\n"
        f"https://t.me/{me.username}?start=ref_{user_id}\n\n"
        "Если помогло — скинь другу 👇",
        reply_markup=get_main_menu(),
    )


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

        # Мини-"вау" перед оплатой: что уже успели сделать за последние минуты
        import time
        now = time.time()
        dq = user_recent_answers.get(user_id)
        if dq:
            while dq and (now - dq[0]) > 120:
                dq.popleft()
        solved = len(dq) if dq else 0
        if solved >= 3:
            saved = solved * 10
            extra = (
                "\n\n🔥 Ты используешь бота как профи.\n"
                f"За последние 2 минуты:\n"
                f"— решено задач: {solved}\n"
                f"— сэкономлено времени: ~{saved} мин\n\n"
                "Хочешь продолжить без ограничений?"
            )
        else:
            extra = ""

        await message.answer(
            PAYWALL_VARIANTS[variant] + extra,
            reply_markup=buy_keyboard(),
        )
        return

    # ВАУ-эффект для первого сообщения
    if int(state.get("prev_used", 0) or 0) == 0:
        await log_event(user_id, "first_message", {"tier": tier})
        await message.answer("⚡ Сейчас покажу, что я умею...")

    if tier != "pro":
        left_after = int(state.get("left_after", 0) or 0)
        last_free_used = (left_after == 0)
        if left_after == 1:
            await log_event(user_id, "free_limit_warning", {"tier": tier, "left": 1})
            await message.answer(
                f"⚠️ Остался 1 бесплатный запрос.\n\n"
                f"Дальше: Basic {TIERS['basic']['price']} ⭐ или Pro {TIERS['pro']['price']} ⭐.\n\n"
                "💡 Нужны Stars? Быстрая покупка:\n"
                "👉 https://t.me/onelinkgo_bot?start=_tgr_KBB2ywM0Mjky"
            )
    else:
        last_free_used = False

    await bot.send_chat_action(message.chat.id, "typing")

    try:
        # Начисляем реф-бонус рефереру после первого реального запроса приглашённого
        # (один раз, без self-ref, без повторов)
        if int(state.get("prev_used", 0) or 0) == 0 and user_data.get("referred_by") and not user_data.get("referral_rewarded"):
            referrer_id = int(user_data.get("referred_by"))
            if referrer_id and referrer_id != user_id:
                def _reward_tx():
                    transaction = db.transaction()
                    referred_ref = _user_ref(user_id)
                    referrer_ref = _user_ref(referrer_id)

                    @firestore.transactional
                    def _run(transaction_obj):
                        referred_snap = referred_ref.get(transaction=transaction_obj)
                        referred_data = referred_snap.to_dict() or {}
                        if referred_data.get("referral_rewarded"):
                            return False
                        if int(referred_data.get("referred_by") or 0) != referrer_id:
                            return False

                        ref_snap = referrer_ref.get(transaction=transaction_obj)
                        ref_data = ref_snap.to_dict() or {}
                        current_bonus = int(ref_data.get("bonus_messages", 0) or 0)
                        transaction_obj.update(referrer_ref, {"bonus_messages": current_bonus + 5})
                        transaction_obj.update(referred_ref, {"referral_rewarded": True})
                        return True

                    return _run(transaction)

                loop = asyncio.get_event_loop()
                rewarded = await loop.run_in_executor(None, _reward_tx)
                if rewarded:
                    await log_event(user_id, "referral_rewarded", {"referrer_id": referrer_id, "bonus": 5})
                    try:
                        await bot.send_message(referrer_id, "🎁 Бонус: +5 сообщений за друга! Спасибо 🙌")
                    except Exception:
                        pass

        user_last_prompt[user_id] = message.text
        answer = await ask_nvidia(user_id, message.text)
        answer = format_answer(answer)

        # Сохраняем последнюю связку вопрос→ответ для фидбека
        import time
        user_last_exchange[user_id] = {
            "prompt": message.text,
            "answer": answer,
            "ts": time.time(),
            "mode": (user_data.get("mode") or "📚 Домашка"),
            "model": user_data.get("model", NVIDIA_MODEL),
        }

        # Запоминаем, что бот реально выдал ответ (для мини-статов paywall)
        user_recent_answers[user_id].append(time.time())

        # Добавляем "ценность" (не в каждом сообщении, чтобы не раздражать)
        current_mode = (user_data.get("mode") or "📚 Домашка")
        if current_mode == "📚 Домашка" and int(state.get("prev_used", 0) or 0) % 2 == 0:
            answer += (
                "\n\n📌 Это типовая задача — такие часто бывают на зачётах/экзаменах.\n"
                "💾 Сохрани решение — пригодится перед контрольной."
            )
        
        # Добавляем микро-продажу в конец
        if tier != "pro":
            answer += "\n\n💡 Хочешь ещё? Напиши следующий запрос"
        
        # Разбиваем длинные сообщения
        parts = split_text(answer)
        for part in parts:
            await message.answer(part)

        # Фидбек 1–5 / "плохо ответил"
        await message.answer("Оцени ответ:", reply_markup=feedback_keyboard())

        # Кнопки "Подробнее" для домашки (чтобы по умолчанию было коротко)
        current_mode = (user_data.get("mode") or "📚 Домашка")
        if current_mode == "📚 Домашка":
            await message.answer("Хочешь подробнее?", reply_markup=homework_details_keyboard())

        # Если это был последний бесплатный ответ — показываем предложение оплаты ПОСЛЕ ответа
        if last_free_used:
            await message.answer(
                "✅ Готово. Это был последний бесплатный ответ.\n\n"
                f"Продолжить:\n"
                f"• Basic — {TIERS['basic']['price']} ⭐: 100 сообщений/30 дней\n"
                f"• Pro — {TIERS['pro']['price']} ⭐: безлимит + все модели\n\n"
                "Нажми кнопку, чтобы открыть доступ 👇",
                reply_markup=buy_keyboard(),
            )
    except Exception as e:
        logger.error(f"Ошибка NVIDIA API: {e}")
        await log_event(user_id, "error_nvidia", {"error": str(e)[:300]})
        await message.answer(
            "⚠️ Сервер загружен, попробуй ещё раз через пару секунд\n\n"
            "Если проблема повторяется — напиши /help"
        )


@dp.message(F.photo)
async def handle_photo(message: Message) -> None:
    user_id = message.from_user.id
    user_data = await get_user(user_id)

    # списываем лимит за обработку фото как за сообщение
    state = await check_access_and_maybe_increment(user_id)
    tier = state.get("tier", user_data.get("tier", "free"))
    if not state.get("allowed", False):
        variant = pick_paywall_variant(user_id)
        await log_event(user_id, "paywall_shown", {"tier": tier, "variant": variant})
        await message.answer(PAYWALL_VARIANTS[variant], reply_markup=buy_keyboard())
        return

    if not message.photo:
        await message.answer("Не вижу фото. Пришли картинку ещё раз.")
        return

    if not OCR_SPACE_API_KEY:
        await message.answer(
            "📸 Я могу решать задачи по фото, но OCR не настроен.\n\n"
            "Чтобы включить распознавание, добавь в Render env переменную `OCR_SPACE_API_KEY`.\n"
            "Пока что: отправь задачу текстом."
        )
        return

    file_id = message.photo[-1].file_id
    tg_file = await bot.get_file(file_id)
    raw = await bot.download_file(tg_file.file_path)
    image_bytes = raw.read() if hasattr(raw, "read") else bytes(raw)

    await message.answer("🔎 Распознаю текст с фото…")
    try:
        text, conf = await ocr_space_image(image_bytes, language="rus")
    except asyncio.TimeoutError:
        await log_event(user_id, "ocr_error", {"error": "timeout"})
        await message.answer("⚠️ Распознавание заняло слишком много времени. Попробуй ещё раз или пришли текстом.")
        return
    except Exception as e:
        await log_event(user_id, "ocr_error", {"error": str(e)[:300]})
        await message.answer("⚠️ Не получилось распознать текст. Попробуй другое фото или пришли текстом.")
        return

    text = (text or "").strip()
    if not text:
        await message.answer("⚠️ На фото не нашёл текста. Попробуй сфоткать ближе/чётче или пришли текстом.")
        return

    user_pending_ocr[user_id] = {"text": text, "conf": conf, "ts": __import__('time').time()}

    if _is_probably_low_quality_ocr(text, conf):
        preview = text[:800]
        await message.answer(
            "Я распознал текст, но качество может быть неидеальным.\n\n"
            f"Текст:\n{preview}\n\n"
            "Использовать его для решения?",
            reply_markup=ocr_confirm_keyboard(),
        )
    else:
        # auto-confirm
        user_last_prompt[user_id] = text
        answer = await ask_nvidia(user_id, text)
        answer = format_answer(answer)
        for part in split_text(answer):
            await message.answer(part)


@dp.message(F.document)
async def handle_document(message: Message) -> None:
    user_id = message.from_user.id
    user_data = await get_user(user_id)

    state = await check_access_and_maybe_increment(user_id)
    tier = state.get("tier", user_data.get("tier", "free"))
    if not state.get("allowed", False):
        variant = pick_paywall_variant(user_id)
        await log_event(user_id, "paywall_shown", {"tier": tier, "variant": variant})
        await message.answer(PAYWALL_VARIANTS[variant], reply_markup=buy_keyboard())
        return

    doc = message.document
    if not doc:
        await message.answer("Не вижу файл. Пришли ещё раз.")
        return

    filename = (doc.file_name or "").lower()
    tg_file = await bot.get_file(doc.file_id)
    raw = await bot.download_file(tg_file.file_path)
    file_bytes = raw.read() if hasattr(raw, "read") else bytes(raw)

    # 1) txt
    if filename.endswith(".txt"):
        try:
            text = file_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            text = ""
        if not text:
            await message.answer("⚠️ Не смог прочитать текст из .txt. Попробуй другой файл.")
            return
        user_last_prompt[user_id] = text
        answer = await ask_nvidia(user_id, text)
        answer = format_answer(answer)
        for part in split_text(answer):
            await message.answer(part)
        return

    # 2) pdf (извлекаем текст, OCR для сканов опционально)
    if filename.endswith(".pdf"):
        await message.answer("📄 Читаю PDF…")
        text = ""
        try:
            text = extract_text_from_pdf(file_bytes)
        except Exception as e:
            await log_event(user_id, "pdf_extract_error", {"error": str(e)[:300]})
            text = ""

        if text and len(text) >= 25:
            user_last_prompt[user_id] = text
            answer = await ask_nvidia(user_id, text)
            answer = format_answer(answer)
            for part in split_text(answer):
                await message.answer(part)
            return

        await message.answer(
            "⚠️ Похоже, это скан без текста (или не смог извлечь текст).\n"
            "Если это картинка в PDF — пришли фото страницы или вставь текст."
        )
        return

    # 3) картинки как document (jpg/png)
    if filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
        if not OCR_SPACE_API_KEY:
            await message.answer(
                "📎 Получил картинку файлом, но OCR не настроен.\n\n"
                "Добавь `OCR_SPACE_API_KEY` в Render env или пришли задачу текстом."
            )
            return
        await message.answer("🔎 Распознаю текст из файла…")
        try:
            text, conf = await ocr_space_image(file_bytes, language="rus")
        except Exception as e:
            await log_event(user_id, "ocr_error", {"error": str(e)[:300]})
            await message.answer("⚠️ Не получилось распознать. Попробуй другой файл или пришли текстом.")
            return
        text = (text or "").strip()
        if not text:
            await message.answer("⚠️ В файле не нашёл текста. Пришли другое изображение или текстом.")
            return
        user_pending_ocr[user_id] = {"text": text, "conf": conf, "ts": __import__('time').time()}
        if _is_probably_low_quality_ocr(text, conf):
            preview = text[:800]
            await message.answer(
                "Я распознал текст, но качество может быть неидеальным.\n\n"
                f"Текст:\n{preview}\n\n"
                "Использовать его для решения?",
                reply_markup=ocr_confirm_keyboard(),
            )
        else:
            user_last_prompt[user_id] = text
            answer = await ask_nvidia(user_id, text)
            answer = format_answer(answer)
            for part in split_text(answer):
                await message.answer(part)
        return

    await message.answer(
        "Я умею читать .txt и .pdf (если PDF содержит текст), и распознавать текст с картинок.\n"
        "Поддержка: .txt, .pdf, .jpg/.png."
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
