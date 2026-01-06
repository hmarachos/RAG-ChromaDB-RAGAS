"""
Скрипт для оценки качества RAG-системы через RAGAS
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rag_assistant import ask_assistant
import config

# Предопределённые вопросы для оценки
EVALUATION_QUESTIONS = [
    "Какие правила работы сервисной службы?",
    "Как восстановить доступ к аккаунту?",
    "Какое время ответа на обращение клиента?",
    "Можно ли использовать продукт на нескольких устройствах?",
    "Как экспортировать данные из системы?"
]


def prepare_dataset(questions: list[str]) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов
    
    Args:
        questions: список вопросов
    
    Returns:
        Dataset для RAGAS
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("Получение ответов от ассистента...")
    
    for i, question in enumerate(questions, 1):
        print(f"  Обработка вопроса {i}/{len(questions)}: {question}")
        
        # Получаем ответ от ассистента
        result = ask_assistant(question)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - объединяем все найденные чанки
        context_texts = [chunk["document"] for chunk in result["context"]]
        contexts_list.append(context_texts)
        
        # Ground truth - для демо оставляем пустым (можно заполнить вручную)
        # RAGAS может работать и без ground truth для некоторых метрик
        ground_truths_list.append("")  # В реальном проекте здесь были бы эталонные ответы
    
    # Создаём датасет
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы
    """
    print("=" * 60)
    print("Оценка качества RAG-системы через RAGAS")
    print("=" * 60)
    
    # Подготавливаем датасет
    dataset = prepare_dataset(EVALUATION_QUESTIONS)
    
    print("\nЗапуск оценки метрик...")
    print("Метрики: faithfulness, answer_relevancy, context_precision")
    
    # Убеждаемся, что переменная окружения установлена
    import os
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
    
    # Настройка эмбеддингов и LLM для RAGAS
    # RAGAS требует использовать обертки для langchain объектов
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        
        # Создаём langchain объекты
        langchain_embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        langchain_llm = ChatOpenAI(
            model_name=config.CHAT_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0
        )
        
        # Обёртываем для RAGAS
        ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        ragas_llm = LangchainLLMWrapper(langchain_llm)
        
        # Создаём метрики с обёрнутыми объектами
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
        
        faithfulness_metric = Faithfulness(llm=ragas_llm)
        answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        
        # ContextPrecision может не принимать embeddings напрямую
        try:
            context_precision_metric = ContextPrecision(llm=ragas_llm, embeddings=ragas_embeddings)
        except TypeError:
            context_precision_metric = ContextPrecision(llm=ragas_llm)
        
        metrics_to_use = [faithfulness_metric, answer_relevancy_metric, context_precision_metric]
        
    except ImportError:
        # Если обёртки недоступны, используем встроенные метрики
        # RAGAS будет использовать переменные окружения
        print("Обёртки RAGAS недоступны, используем встроенные метрики с переменными окружения")
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
    except Exception as e:
        # Если что-то пошло не так, используем встроенные метрики
        print(f"Используем встроенные метрики (предупреждение: {e})")
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]
    
    # Запускаем оценку
    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_use
    )
    
    # Выводим результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 60)
    
    # RAGAS возвращает Dataset с результатами
    # Вычисляем средние значения метрик, игнорируя nan
    import math
    
    faithfulness_values = [v for v in result['faithfulness'] if not math.isnan(v)] if result['faithfulness'] else []
    answer_relevancy_values = [v for v in result['answer_relevancy'] if not math.isnan(v)] if result['answer_relevancy'] else []
    context_precision_values = [v for v in result['context_precision'] if not math.isnan(v)] if result['context_precision'] else []
    
    avg_faithfulness = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
    avg_answer_relevancy = sum(answer_relevancy_values) / len(answer_relevancy_values) if answer_relevancy_values else float('nan')
    avg_context_precision = sum(context_precision_values) / len(context_precision_values) if context_precision_values else 0
    
    # Выводим общие метрики
    print(f"\nFaithfulness (верность ответа): {avg_faithfulness:.4f}")
    if not math.isnan(avg_answer_relevancy):
        print(f"Answer Relevancy (релевантность ответа): {avg_answer_relevancy:.4f}")
    else:
        print(f"Answer Relevancy (релевантность ответа): не удалось вычислить (ошибка с эмбеддингами)")
    print(f"Context Precision (точность контекста): {avg_context_precision:.4f}")
    
    # Выводим детали по каждому вопросу
    print("\n" + "=" * 60)
    print("ДЕТАЛИ ПО ВОПРОСАМ")
    print("=" * 60)
    
    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\nВопрос {i+1}: {question}")
        print(f"  Faithfulness: {result['faithfulness'][i]:.4f} ---точность ответа")
        ar_val = result['answer_relevancy'][i]
        if not math.isnan(ar_val):
            print(f"  Answer Relevancy: {ar_val:.4f} ---релевантность ответа вопросу")
        else:
            print(f"  Answer Relevancy: не удалось вычислить ---релевантность ответа вопросу")
        print(f"  Context Precision: {result['context_precision'][i]:.4f} ---точность выбранного контекста")
    
    print("\n" + "=" * 60)
    print("Оценка завершена!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_rag_system()

