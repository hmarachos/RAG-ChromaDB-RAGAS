"""
Скрипт для загрузки документов, создания эмбеддингов и сохранения в ChromaDB
"""
import os
import re
from pathlib import Path
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import config

# Инициализация OpenAI клиента
client = OpenAI(api_key=config.OPENAI_API_KEY)


def clean_text(text: str) -> str:
    """
    Очистка текста: удаление лишних пробелов и переносов строк
    """
    # Удаляем множественные пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    # Удаляем пробелы в начале и конце
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Разбиение текста на чанки с overlap
    
    Args:
        text: исходный текст
        chunk_size: размер чанка в символах
        overlap: размер overlap между чанками в символах
    
    Returns:
        список чанков
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Определяем конец текущего чанка
        end = start + chunk_size
        
        # Если это не последний чанк, пытаемся разбить по границе слова
        if end < len(text):
            # Ищем ближайший пробел перед концом чанка
            space_pos = text.rfind(' ', start, end)
            if space_pos != -1:
                end = space_pos
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Перемещаемся на следующий чанк с учетом overlap
        start = end - overlap
        if start <= 0:
            start = end
    
    return chunks


def get_embedding(text: str) -> list[float]:
    """
    Получение эмбеддинга для текста через OpenAI
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def ingest_documents():
    """
    Основная функция индексации документов
    """
    print("Начало индексации документов...")
    
    # Инициализация ChromaDB
    # Если коллекция существует, удаляем её для чистой индексации
    chroma_client = chromadb.PersistentClient(
        path=config.CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Удаляем существующую коллекцию, если она есть
    try:
        chroma_client.delete_collection("rag_collection")
    except:
        pass
    
    # Создаём новую коллекцию
    collection = chroma_client.create_collection(
        name="rag_collection",
        metadata={"description": "RAG collection for demo project"}
    )
    
    # Загружаем все txt файлы из папки data
    data_path = Path(config.DATA_DIR)
    txt_files = list(data_path.glob("*.txt"))
    
    if not txt_files:
        print(f"Не найдено txt файлов в папке {config.DATA_DIR}")
        return
    
    print(f"Найдено {len(txt_files)} файлов для индексации")
    
    all_chunks = []
    all_embeddings = []
    all_metadatas = []
    all_ids = []
    
    chunk_counter = 0
    
    for file_path in txt_files:
        print(f"Обработка файла: {file_path.name}")
        
        # Читаем файл
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Очищаем текст
        cleaned_text = clean_text(text)
        
        # Разбиваем на чанки
        chunks = chunk_text(cleaned_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print(f"  Создано {len(chunks)} чанков")
        
        # Создаём эмбеддинги для каждого чанка
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            
            all_chunks.append(chunk)
            all_embeddings.append(embedding)
            all_metadatas.append({
                "source": file_path.name,
                "chunk_id": i
            })
            all_ids.append(f"{file_path.stem}_chunk_{i}")
            
            chunk_counter += 1
            if chunk_counter % 10 == 0:
                print(f"  Обработано {chunk_counter} чанков...")
    
    # Сохраняем все чанки в ChromaDB одной операцией
    print(f"\nСохранение {len(all_chunks)} чанков в ChromaDB...")
    collection.add(
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    print(f"Индексация завершена! Всего чанков: {len(all_chunks)}")
    print(f"ChromaDB сохранена в: {config.CHROMA_DB_PATH}")


if __name__ == "__main__":
    ingest_documents()

