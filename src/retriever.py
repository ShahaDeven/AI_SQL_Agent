import json
import shutil
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document 

JSON_PATH = "D:\\Main_Python\\Projects\\AI_SQL_Agent\data\\sql_examples.json"
CHROMA_PATH = "D:\\Main_Python\\Projects\\AI_SQL_Agent\data\\chroma_db"

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

    print(f"🔄 Vectorizing {len(documents)} examples... (This uses a free local model)")
    
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
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