---
inclusion: always
---

# Проект: AI Telegram-бот на NVIDIA NIM

## Стек
- Python 3.10+
- aiogram 3.7 (async Telegram Bot framework)
- openai SDK (для NVIDIA NIM API — совместимый интерфейс)
- python-dotenv (конфигурация через .env)
- firebase-admin SDK (Firestore — база данных)

## Структура проекта
```
ai bot/
├── bot.py            # весь код бота (один файл)
├── .env              # токены и настройки (не коммитить!)
├── firebase.json     # сервисный аккаунт Firebase (не коммитить!)
├── requirements.txt  # зависимости
└── .kiro/steering/   # steering-файлы для Kiro
```

## Правила разработки
- Весь код бота — в одном файле `bot.py`
- Все настройки — только через `.env`, никаких хардкодов токенов
- Хранилища данных — Firebase Firestore (данные сохраняются между перезапусками)
- Новые модели добавлять в словарь `AVAILABLE_MODELS` в `bot.py`
- Обработчики регистрировать через декораторы aiogram 3 (`@dp.message`, `@dp.callback_query`)
- Все запросы к NVIDIA API — через `ask_nvidia()`, не напрямую
- Синхронные вызовы SDK оборачивать в `loop.run_in_executor()`

## Переменные окружения (.env)
| Переменная      | Обязательная | По умолчанию                      | Описание                        |
|-----------------|:------------:|-----------------------------------|---------------------------------|
| FIREBASE_CREDENTIALS|              | firebase.json                     | Путь к JSON сервисного аккаунта |
| TELEGRAM_TOKEN  | ✅           | —                                 | Токен бота от @BotFather        |
| NVIDIA_API_KEY  | ✅           | —                                 | API ключ с build.nvidia.com     |
| NVIDIA_MODEL    |              | meta/llama-3.1-70b-instruct       | Модель по умолчанию             |
| ADMIN_USERNAME  |              | @your_username                    | Юзернейм для связи по оплате    |
| FREE_LIMIT      |              | 5                                 | Кол-во бесплатных сообщений     |
| MAX_HISTORY     |              | 20                                | Глубина истории диалога         |
| MAX_TOKENS      |              | 1024                              | Макс. токенов в ответе          |
| TEMPERATURE     |              | 0.7                               | Температура генерации (0.0–1.0) |

## NVIDIA NIM API
- Base URL: `https://integrate.api.nvidia.com/v1`
- Совместим с OpenAI SDK (`openai.OpenAI(base_url=..., api_key=...)`)
- Каталог моделей: https://build.nvidia.com/models
- Free Endpoint модели не требуют оплаты на стороне NVIDIA
