# 🤖 AskNeuro AI — Telegram Bot

AI-бот на базе NVIDIA NIM API с монетизацией через Telegram Stars и Firebase Firestore.

## ✨ Возможности

- 🧠 7 бесплатных NVIDIA NIM моделей (Llama, Nemotron, Mistral, Qwen, GLM, MiniMax)
- 💸 Монетизация через Telegram Stars (XTR)
- 🎯 3 тарифа: Free (5 сообщений) / Basic (100/месяц, 100⭐) / Pro (безлимит, 200⭐)
- 📚 4 режима работы: Домашка, Заработок, Тексты, TikTok идеи
- 🔥 Firebase Firestore — история диалогов и данные пользователей
- 👨‍💼 Админ-панель: статистика, рассылка, бан/разбан
- 🎨 Красивое форматирование с эмодзи (без markdown)
- 📊 Продающая воронка для максимальной конверсии

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка `.env`

Создай файл `.env` в корне проекта:

```env
# Telegram
TELEGRAM_TOKEN=your_bot_token_from_botfather

# NVIDIA NIM API
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_MODEL=meta/llama-3.1-70b-instruct

# Firebase
FIREBASE_CREDENTIALS=firebase.json

# Настройки бота
FREE_LIMIT=5
ADMIN_USERNAME=@your_username
ADMIN_ID=your_telegram_user_id
STARS_PRICE=100
MAX_HISTORY=20
MAX_TOKENS=1024
TEMPERATURE=0.7
```

### 3. Настройка Firebase

1. Создай проект в [Firebase Console](https://console.firebase.google.com/)
2. Включи Firestore Database (Native mode)
3. Скачай сервисный аккаунт: Project Settings → Service Accounts → Generate new private key
4. Сохрани как `firebase.json` в корне проекта

### 4. Запуск

```bash
python bot.py
```

## 📦 Структура проекта

```
ai bot/
├── bot.py              # Весь код бота
├── .env                # Конфигурация (не коммитить!)
├── firebase.json       # Firebase credentials (не коммитить!)
├── requirements.txt    # Зависимости Python
├── .gitignore         # Игнорируемые файлы
├── README.md          # Документация
└── .kiro/steering/    # Steering-файлы для Kiro AI
    ├── project.md
    ├── done.md
    └── roadmap.md
```

## 🎯 Команды бота

### Пользовательские
- `/start` — Начало работы с ботом
- `/help` — Справка по тарифам и моделям
- `/status` — Статус аккаунта и остаток сообщений
- `/model` — Выбор AI модели
- `/clear` — Очистка истории диалога

### Админские (требуют ADMIN_ID)
- `/admin` — Статистика бота
- `/grant <user_id>` — Выдать доступ пользователю
- `/ban <user_id>` — Заблокировать пользователя
- `/unban <user_id>` — Разблокировать пользователя
- `/broadcast <текст>` — Рассылка всем пользователям

## 🎨 Режимы работы

- 📚 **Домашка** — Помощь с учёбой, решение задач
- 💸 **Заработок** — Идеи заработка с планом действий
- ✍️ **Тексты** — Профессиональный копирайтинг
- 🎬 **TikTok идеи** — Вирусные сценарии для TikTok

## 💰 Тарифы

| Тариф | Лимит | Модели | Цена |
|-------|-------|--------|------|
| Free | 5 сообщений | 2 базовые | Бесплатно |
| Basic | 100/месяц | 4 модели | 100 ⭐ |
| Pro | Безлимит | 7 моделей | 200 ⭐ |

## 🔧 Технологии

- **Python 3.10+**
- **aiogram 3.7** — Async Telegram Bot framework
- **openai SDK** — Для NVIDIA NIM API
- **firebase-admin** — Firestore база данных
- **python-dotenv** — Управление конфигурацией

## 📝 Firestore структура

### Коллекция `users/{user_id}`
```json
{
  "messages_used": 0,
  "paid": false,
  "tier": "free",
  "model": "meta/llama-3.1-70b-instruct",
  "username": "user123",
  "banned": false,
  "last_reset": "2026-03-24T12:00:00Z"
}
```

### Коллекция `history/{user_id}/messages/{doc_id}`
```json
{
  "role": "user",
  "content": "Привет!",
  "ts": "2026-03-24T12:00:00Z"
}
```

## 🚀 Деплой

### Docker (скоро)
```bash
docker-compose up -d
```

### VPS (Ubuntu)
```bash
# Установка зависимостей
sudo apt update
sudo apt install python3-pip python3-venv

# Создание venv
python3 -m venv venv
source venv/bin/activate

# Установка пакетов
pip install -r requirements.txt

# Запуск через systemd
sudo systemctl enable aibot
sudo systemctl start aibot
```

## 📊 Roadmap

- [x] Базовый AI-бот с NVIDIA NIM
- [x] Лимиты и монетизация
- [x] Telegram Stars оплата
- [x] Firebase Firestore
- [x] Админ-панель
- [x] Тарифы (Free/Basic/Pro)
- [x] Красивое форматирование
- [x] Кнопочное меню с режимами
- [x] Продающая воронка
- [ ] Голосовые сообщения (Whisper API)
- [ ] Docker + docker-compose
- [ ] Webhook режим
- [ ] Генерация изображений (FLUX)
- [ ] Реферальная система

## 📄 Лицензия

MIT License

## 👨‍💻 Автор

Создано с помощью [Kiro AI](https://kiro.ai)

## 🤝 Поддержка

Если возникли вопросы — открой Issue на GitHub!

## 🚀 Запуск и первые продажи

Смотри чеклист и 20 готовых демо-кейсов в `LAUNCH.md`.
