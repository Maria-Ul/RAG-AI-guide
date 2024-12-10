from langchain.schema import Document
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from multiprocessing import Pool
import time
import os
hf_token = os.getenv("HF_TOKEN")

def get_retrieval_chain(retriever, document_chain):
    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    ).assign(
        answer=document_chain,
    )
    return retrieval_chain


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
    dataset = load_dataset("legacy-datasets/wikipedia", "20220301.simple", trust_remote_code=True,  download_mode="force_redownload")  # русский пока не работает
    train_dataset = [Document(page_content=entry) for entry in dataset['train']['text'][:100]]
    return train_dataset


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


def process_data(train_dataset, embedding_model_name, chunk_size, chunk_overlap_scale):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True}
    )
    print("init model ")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],  # separates either on words or paragraphs
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * chunk_overlap_scale,
    )
    print("init splitter ")
    all_splits = splitter.split_documents(train_dataset)

    print("init all_splits ")
    try:
        faiss_cosine = FAISS.from_documents(all_splits, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
    # faiss_cosine = FAISS.from_documents(all_splits, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    # print("faiss is not none")
    # if faiss_cosine is None:
    #     raise ValueError("faiss_cosine is None, check the FAISS creation process")

    return faiss_cosine


def get_retriever(train_dataset):
    print('start get_retriever')
    top_k = 4
    embedding_model_name = 'cointegrated/rubert-tiny2'
    chunk_size = 350
    chunk_overlap_scale = 0.1

    # Переносим использование multiprocessing в блок __main__
    faiss_cosine = None
    # if __name__ == '__main__':
    with Pool() as pool:
        print('start pool')
        faiss_cosine = pool.apply(process_data, args=(train_dataset, embedding_model_name, chunk_size, chunk_overlap_scale))
        print('get  faiss')
    return faiss_cosine.as_retriever(k=top_k)


def chat(user_input, retrieval_chain, history=False):
    print(f"User input: {user_input}")
    chat_history = [
        SystemMessage(content="You're a useful assistant. Follow the user's commands, clarify the missing data.")
    ]

    try:
        result = retrieval_chain.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ],
            }
        )
        print(f"Result from retrieval chain: {result}")
        if history:
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=result["answer"]))
        print(f"result : {result['answer']} type {type(result['answer'])}")
        print(f"Bot: {result['answer']}")
        return result['answer']
    except Exception as e:
        print(f"Error: {str(e)}")
    # print(f"result : {result['answer']} type {type(result['answer'])}")
    # return result['answer']
