# from flask import Flask, request, jsonify
# from src.rag.rag_guide import get_chain, get_data, get_retriever, get_retrieval_chain, chat
#
# app = Flask(__name__)
#
# # Инициализация RAG-агента
# def init_rag_agent():
#     # Получаем данные, retriever, цепочку документов, цепочку поиска
#     train_dataset = get_data()
#     print("Get data")
#     retriever = get_retriever(train_dataset)
#     print("Get retriever")
#     document_chain = get_chain()
#     print("Get chain")
#     retrieval_chain = get_retrieval_chain(retriever, document_chain)
#     print("Get retrieval_chain")
#     return retrieval_chain
#
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
#         # Получаем retrieval_chain для генерации ответа
#         retrieval_chain = init_rag_agent()
#
#         # Используем RAG-агент для ответа
#         print("Calling chat function with the question...")
#         answer = chat(user_question, retrieval_chain=retrieval_chain)
#         print(f"Generated answer: {answer}")
#         # Возвращаем ответ в формате JSON
#         return jsonify({"answer": answer})
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == '__main__':
#     # Запуск приложения Flask
#     app.run(host="0.0.0.0", port=8000, debug=True)

from flask import Flask, request, jsonify
from src.rag.rag_guide import get_data, get_retriever, get_chain, get_retrieval_chain, chat

app = Flask(__name__)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    try:
        # Получение данных из запроса
        data = request.get_json()
        print(f"Received data: {data}")
        user_question = data.get("question", "")

        if not user_question:
            return jsonify({"error": "Question is required"}), 400

        # Используем RAG-агент для ответа
        answer = chat(user_question, retrieval_chain=retrieval_chain)
        print(f"Generated answer: {answer}")

        # Возвращаем ответ в формате JSON
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


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

    # Запуск Flask-приложения
    app.run(host="0.0.0.0", port=8000)
