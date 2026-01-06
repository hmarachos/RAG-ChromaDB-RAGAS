"""
RAG-ассистент: поиск релевантного контекста и генерация ответов
"""
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import config

# Инициализация клиентов
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(
    path=config.CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Получаем коллекцию
collection = chroma_client.get_collection("rag_collection")


def get_embedding(text: str) -> list[float]:
    """
    Получение эмбеддинга для текста через OpenAI
    """
    response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def search_relevant_chunks(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """
    Поиск релевантных чанков в ChromaDB
    
    Args:
        query: вопрос пользователя
        top_k: количество возвращаемых чанков
    
    Returns:
        список словарей с полями: document, metadata, distance
    """
    # Создаём эмбеддинг для запроса
    query_embedding = get_embedding(query)
    
    # Ищем релевантные чанки
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Формируем результат в удобном формате
    chunks = []
    if results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            chunks.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
    
    return chunks


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Генерация ответа на основе контекста через OpenAI
    
    Args:
        query: вопрос пользователя
        context_chunks: список релевантных чанков
    
    Returns:
        ответ ассистента
    """
    # Собираем контекст из чанков
    context = "\n\n".join([
        f"[Источник: {chunk['metadata']['source']}]\n{chunk['document']}"
        for chunk in context_chunks
    ])
    
    # Системная инструкция
    system_prompt = "Ты — полезный ассистент. Отвечай строго на основе предоставленного контекста. Если в контексте нет информации для ответа, так и скажи."
    
    # Промпт с контекстом
    user_prompt = f"""Контекст:
{context}

Вопрос: {query}

Ответ:"""
    
    # Отправляем запрос в OpenAI
    response = openai_client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1  # Низкая температура для более точных ответов
    )
    
    return response.choices[0].message.content


def ask_assistant(query: str) -> dict:
    """
    Основная функция ассистента: поиск контекста и генерация ответа
    
    Args:
        query: вопрос пользователя
    
    Returns:
        словарь с ответом и контекстом
    """
    # Ищем релевантные чанки
    chunks = search_relevant_chunks(query)
    
    if not chunks:
        return {
            "answer": "Извините, не удалось найти релевантную информацию в базе знаний.",
            "context": []
        }
    
    # Генерируем ответ
    answer = generate_answer(query, chunks)
    
    return {
        "answer": answer,
        "context": chunks
    }


def main():
    """
    CLI интерфейс для взаимодействия с ассистентом
    """
    print("=" * 60)
    print("RAG-ассистент готов к работе!")
    print("Введите ваш вопрос (или 'exit' для выхода)")
    print("=" * 60)
    
    while True:
        query = input("\nВопрос: ").strip()
        
        if query.lower() in ['exit', 'quit', 'выход']:
            print("До свидания!")
            break
        
        if not query:
            continue
        
        print("\nОбработка запроса...")
        result = ask_assistant(query)
        
        print("\n" + "=" * 60)
        print("Ответ:")
        print(result["answer"])
        print("=" * 60)
        
        # Опционально показываем источники
        if result["context"]:
            print(f"\nНайдено релевантных чанков: {len(result['context'])}")
            show_sources = input("Показать источники? (y/n): ").strip().lower()
            if show_sources == 'y':
                for i, chunk in enumerate(result["context"], 1):
                    print(f"\n--- Источник {i} ---")
                    print(f"Файл: {chunk['metadata']['source']}")
                    print(f"Чанк ID: {chunk['metadata']['chunk_id']}")
                    print(f"Текст: {chunk['document'][:200]}...")


if __name__ == "__main__":
    main()

