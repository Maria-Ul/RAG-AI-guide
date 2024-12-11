from flask import Flask, request, jsonify
from src.rag.rag_guide import get_data, get_retriever, get_chain, get_retrieval_chain, chat

app = Flask(__name__)

# @app.route('/generate_text', methods=['POST'])
# def generate_text():
#     try:
#         # Получение данных из запроса
#         data = request.get_json()
#         print(f"Received data: {data}")
#         user_question = data.get("question", "")
#
#         if not user_question:
#             return jsonify({"error": "Question is required"}), 400
#
#         # Используем RAG-агент для ответа
#         retrieval_chain = app.config['retrieval_chain']
#         answer = chat(user_question, retrieval_chain=retrieval_chain)
#         print(f"Generated answer: {answer}")
#
#         # Убедимся, что возвращаем только текст
#         if isinstance(answer, str):
#             clean_answer = answer.strip('"')  # Убираем лишние кавычки
#             return jsonify({"answer": clean_answer})
#         else:
#             print("Unexpected answer format:", answer)
#             return jsonify({"error": "Invalid response format"}), 500
#
#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({"error": str(e)}), 500

@app.route('/generate_text', methods=['POST'])
def generate_text():
    try:
        # Получение данных из запроса
        data = request.get_json()
        print(f"Received data: {data}")
        user_question = data.get("question", "")

        if not user_question:
            return jsonify({"error": "Question is required"}), 400

        # Генерация ответа через RAG-агент
        print("Calling chat function with the question...")
        answer = chat(user_question, retrieval_chain=retrieval_chain)
        print(f"Generated answer: {answer}")

        # Убедимся, что возвращаем только текст
        if isinstance(answer, str):
            clean_answer = answer.strip('"')  # Убираем лишние кавычки
            return jsonify({"answer": clean_answer})
        else:
            print("Unexpected answer format:", answer)
            return jsonify({"error": "Invalid response format"}), 500

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Sorry, something went wrong."}), 500


if __name__ == '__main__':
    # Инициализация RAG-агента
    print('Initializing RAG agent...')
    train_dataset = get_data()
    print('Dataset loaded')

    retriever = get_retriever(train_dataset)
    print('Retriever initialized')

    document_chain = get_chain()
    print('Document chain created')

    retrieval_chain = get_retrieval_chain(retriever, document_chain)
    print('Retrieval chain ready')
    app.config['retrieval_chain'] = retrieval_chain

    # Запуск Flask-приложения
    app.run(host="0.0.0.0", port=8000)
