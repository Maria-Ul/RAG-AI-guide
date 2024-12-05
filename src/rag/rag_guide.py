from langchain.schema import Document
from datasets import load_dataset, Dataset
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict

from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain.schema import HumanMessage, SystemMessage, AIMessage
import time


# def answer_query(query):
#     # Получаем эмбеддинг для запроса
#     retriever = faiss_cosine
#     docs = retriever.similarity_search(query, k=3)
#     return [doc.page_content for doc in docs]
#
# # Пример использования
# user_query = "Что можно посмотреть в Париже?"
# answers = answer_query(user_query)
# print("Ответы:", answers)

def get_chain():
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    chat_model = ChatHuggingFace(llm=llm)

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context.
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    <context>
    {context}
    </context>
    """
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    document_chain = create_stuff_documents_chain(chat_model, question_answering_prompt)
    return document_chain


def get_data():
    dataset = load_dataset("legacy-datasets/wikipedia", "20220301.simple")  # русский пока не работает
    train_dataset = [Document(page_content=entry) for entry in dataset['train']['text'][:100]]
    return train_dataset


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


def get_retriever(train_dataset):
    TOP_K = 4
    EMBEDDING_MODEL_NAME = 'cointegrated/rubert-tiny2'
    CHUNK_SIZE = 350
    CHUNK_OVERLAP_SCALE = 0.1
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True}
    )
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],  # separates either on words or paragraphs
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE * CHUNK_OVERLAP_SCALE,
    )
    all_splits = splitter.split_documents(train_dataset)
    faiss_cosine = FAISS.from_documents(all_splits, embedding_model, distance_strategy=DistanceStrategy.COSINE)

    return faiss_cosine.as_retriever(k=TOP_K)


def chat(retrieval_chain, history=False):
    chat_history = [
        SystemMessage(content="You're a useful assistant. Follow the user's commands, clarify the missing data.")
    ]

    while True:
        user_input = input("User: ")
        if user_input == "":
            break
        # print(f"User: {user_input}")
        try:
            result = retrieval_chain.invoke(
              {
                  # "chat_history": chat_history,
                  "messages": [
                      HumanMessage(content=user_input)
                  ],
              }
            )
            if history:
              chat_history.append(HumanMessage(content=user_input))
              chat_history.append(AIMessage(content=result["answer"]))
            print(f"Bot: {result['answer']}")
        except Exception as e:
            print(f"Error: {str(e)}")
        time.sleep(0.2)


if __name__ == "__main__":
    train_dataset = get_data()
    retriever = get_retriever(train_dataset)
    document_chain = get_chain()

    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    ).assign(
        answer=document_chain,
    )

    chat(retrieval_chain, history=False)

