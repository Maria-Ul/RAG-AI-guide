# -*- coding: utf-8 -*-
"""rag_guide.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10AHM7izyZ0MjydOJ5LGqpFDJWtmqL1cm
"""

# !pip install faiss-gpu
#
# !pip install datasets #faiss-cpu
#
# !pip install pandas numpy seaborn matplotlib scikit-learn spacy sentence_transformers langchain langchain_community langchain_chroma langchain_openai
#
# !pip install mwparserfromhell

from langchain.schema import Document
from datasets import load_dataset, Dataset
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_NAME = 'cointegrated/rubert-tiny2'
CHUNK_SIZE = 350
CHUNK_OVERLAP_SCALE = 0.1
TOP_K = 3

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "], # separates either on words or paragraphs
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_SIZE*CHUNK_OVERLAP_SCALE,
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    encode_kwargs={"normalize_embeddings": True}
)

"""https://huggingface.co/datasets/legacy-datasets/wikipedia"""

dataset = load_dataset("legacy-datasets/wikipedia", "20220301.simple") #русский пока не работает

# dataset = load_dataset("stepkurniawan/sustainability-methods-wiki", "50_QA_reviewed")

dataset

train_dataset = [Document(page_content=entry) for entry in dataset['train']['text'][:100]]
train_dataset[:1]

all_splits = splitter.split_documents(train_dataset)

# faiss_eucledian = FAISS.from_documents(all_splits, embedding_model, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
faiss_cosine = FAISS.from_documents(all_splits, embedding_model, distance_strategy=DistanceStrategy.COSINE)
# faiss_ip = FAISS.from_documents(all_splits, embedding_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

def answer_query(query):
    # Получаем эмбеддинг для запроса
    query_embedding = embedding_model.embed_query(query)
    retriever = faiss_cosine
    docs = retriever.similarity_search(query, k=3)
    return [doc.page_content for doc in docs]

# Пример использования
user_query = "Что можно посмотреть в Париже?"
answers = answer_query(user_query)
print("Ответы:", answers)

