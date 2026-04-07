# Firebase MCP Server Configuration

## Installation

### Via Smithery (recommended)
```bash
npx @smithery/cli install @smithery-ai/firebase --client claude
```

### Manual Installation
1. Убедись, что установлен Node.js 18+
2. Клонируй репозиторий:
   ```bash
   git clone https://github.com/smithery-ai/firebase.git
   cd firebase
   npm install
   npm run build
   ```

## Environment Variables
```env
FIREBASE_PROJECT_ID=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=path/to/serviceAccount.json
# или
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
```

## Возможности
- Чтение/запись Firestore документов
- Управление Firebase Auth пользователями
- Запуск Cloud Functions
- Доступ к Firebase Storage
- Firebase Analytics данные
