import json
import shutil
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) 

JSON_PATH = os.path.join(PROJECT_ROOT, "data", "sql_examples.json")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

print("Loading Local Embedding Model... (This happens ONCE)")
embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} 
)

def _load_documents():
    """
    Helper to load raw documents from JSON.
    Used by both the Vector DB setup and the BM25 Retriever.
    """
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON file not found at {JSON_PATH}")

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    documents = []
    for entry in data:
        doc = Document(
            page_content=entry['question'], 
            metadata={"sql_query": entry['sql']}
        )
        documents.append(doc)
    return documents

def setup_vector_db():
    """
    Reads the JSON file and creates a persistent Vector Database.
    Run this ONCE to train the system.
    """
    documents = _load_documents()

    print(f"Vectorizing {len(documents)} examples locally...")
    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function, 
        persist_directory=CHROMA_PATH
    )
    print("Local Vector Database created at data/chroma_db")
    return vector_db

def get_few_shot_examples(user_query, k=3):
    """
    Retrieves examples using HYBRID SEARCH (Vector + Keyword).
    """
    documents = _load_documents()
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k 

    vector_db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_function
    )
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    results = ensemble_retriever.invoke(user_query)

    formatted_examples = ""
    for i, doc in enumerate(results):
        formatted_examples += f"Example {i+1}:\n"
        formatted_examples += f"User Q: {doc.page_content}\n"
        formatted_examples += f"SQL: {doc.metadata['sql_query']}\n\n"
        
    return formatted_examples

if __name__ == "__main__":
    setup_vector_db()
    
    test_q = "How many high risk customers do we have?"
    print(f"\nTesting Hybrid Retrieval for: '{test_q}'")
    print("-" * 30)
    print(get_few_shot_examples(test_q))