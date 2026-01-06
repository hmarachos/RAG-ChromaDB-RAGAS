"""
Конфигурация проекта RAG с ChromaDB и RAGAS
"""
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# OpenAI API ключ (читается из переменной окружения)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не установлен в переменных окружения")

# Модель для эмбеддингов
EMBEDDING_MODEL = "text-embedding-3-small"

# Модель для чата
CHAT_MODEL = "gpt-3.5-turbo"

# Путь к ChromaDB
CHROMA_DB_PATH = "./chroma_db"

# Параметры чанкинга
CHUNK_SIZE = 500  # размер чанка в символах
CHUNK_OVERLAP = 100  # overlap между чанками в символах

# Параметры поиска
TOP_K = 5  # количество релевантных чанков для поиска

# Путь к данным
DATA_DIR = "./data"

