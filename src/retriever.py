import json
import shutil
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document 
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ GOOGLE_API_KEY not found! Make sure it is in your .env file.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) 

JSON_PATH = os.path.join(PROJECT_ROOT, "data", "sql_examples.json")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def setup_vector_db():
    """
    Reads the JSON file and creates a persistent Vector Database.
    Run this ONCE to train the system.
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

    print(f"🔄 Vectorizing {len(documents)} examples...")
    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    print("Vector Database created at data/chroma_db")
    return vector_db

def get_few_shot_examples(user_query, k=3):
    """
    Retrieves the top K most similar SQL examples for a given question.
    """
    
    vector_db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_function
    )

    results = vector_db.similarity_search(user_query, k=k)

    formatted_examples = ""
    for i, doc in enumerate(results):
        formatted_examples += f"Example {i+1}:\n"
        formatted_examples += f"User Q: {doc.page_content}\n"
        formatted_examples += f"SQL: {doc.metadata['sql_query']}\n\n"
        
    return formatted_examples

if __name__ == "__main__":
    setup_vector_db()
    
    test_q = "How many high risk customers do we have?"
    print(f"\nTesting Retrieval for: '{test_q}'")
    print("-" * 30)
    print(get_few_shot_examples(test_q))