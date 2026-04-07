# Render MCP Server Configuration

## Installation

### Via Smithery (recommended)
```bash
npx @smithery/cli install @smithery-ai/render --client claude
```

### Manual Installation
1. Убедись, что установлен Node.js 18+
2. Клонируй репозиторий:
   ```bash
   git clone https://github.com/smithery-ai/render.git
   cd render
   npm install
   npm run build
   ```

## Environment Variables
```env
RENDER_API_KEY=your_render_api_key
```

## Получение API ключа Render
1. Зайди в Render Dashboard: https://dashboard.render.com
2. Account Settings → API Keys
3. Создай новый ключ и скопируй в `.env`

## Возможности
- Получение статуса сервисов
- Перезапуск web services
- Просмотр логов
- Управление deploy'ами
