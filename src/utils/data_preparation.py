from langchain.schema import Document
from datasets import load_dataset, Dataset
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_data():
    dataset = load_dataset("legacy-datasets/wikipedia", "20220301.simple")  # русский пока не работает
    texts = [Document(page_content=entry) for entry in dataset['train']['text']]
    return texts
def get_retriever(train_dataset):
    EMBEDDING_MODEL_NAME = 'cointegrated/rubert-tiny2'
    CHUNK_SIZE = 350
    CHUNK_OVERLAP_SCALE = 0.1
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True}
    )
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE * CHUNK_OVERLAP_SCALE,
    )
    all_splits = splitter.split_documents(train_dataset)

    FAISS_INDEX_FILE = "my_faiss_index.faiss"

    faiss_index = FAISS.from_documents(
        all_splits,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE
    )
    faiss_index.save_local(FAISS_INDEX_FILE)
