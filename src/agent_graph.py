import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import duckdb
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.retriever import get_few_shot_examples
import sqlparse
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Define the paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) 

DEMO_DB_PATH = os.path.join(PROJECT_ROOT, "data", "sql_agent_demo.db")
FULL_DB_PATH = os.path.join(PROJECT_ROOT, "data", "supply_chain.db")

if os.path.exists(DEMO_DB_PATH):
    DB_PATH = DEMO_DB_PATH
    print(f"Using DEMO database: {DB_PATH}")
elif os.path.exists(FULL_DB_PATH):
    DB_PATH = FULL_DB_PATH
    print(f"Using FULL database: {DB_PATH}")
else:
    raise FileNotFoundError(f"Critical Error: No database found! Checked: \n1. {DEMO_DB_PATH}\n2. {FULL_DB_PATH}")

MODEL_NAME = "gemini-2.5-flash" 
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, transport="rest")
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

CACHE_FILE = os.path.join(PROJECT_ROOT, "sql_cache.json")

def get_schema():
    """
    Returns the database schema to the LLM so it knows table names.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    schema_str = "Database Schema (DuckDB):\n"
    target_tables = ['customer', 'lineitem', 'orders', 'supplier', 'nation', 'part']
    
    for table in target_tables:
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        col_list = [f"{col[0]} {col[1]}" for col in cols]
        schema_str += f"- {table}: {', '.join(col_list)}\n"
        
    con.close()
    return schema_str

def get_cached_sql(question):
    """Checks if we have already answered this exact question."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            return cache.get(question)
        except:
            return None
    return None

def save_to_cache(question, sql):
    """Saves valid SQL to the cache file."""
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
        except:
            cache = {}
    
    cache[question] = sql
    
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def run_query(sql_query):
    """
    Executes SQL. Returns (Result, Error).
    """
    try:
        clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()
        
        con = duckdb.connect(DB_PATH, read_only=True)
        if "DROP" in clean_sql.upper() or "DELETE" in clean_sql.upper():
            return None, "Security Error: Read-Only Access."
            
        df = con.execute(sql_query).fetchdf()
        con.close()
        return df, None
    except Exception as e:
        return None, str(e)

def check_sql_safety(sql_query):
    BLOCKED_KEYWORDS = {'DELETE', 'UPDATE', 'DROP', 'INSERT', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE'}
    
    parsed = sqlparse.parse(sql_query)
    for statement in parsed:
        stmt_type = statement.get_type()
        if stmt_type and stmt_type.upper() not in ['SELECT', 'UNKNOWN']:
            raise ValueError(f"SECURITY ALERT: Harmful SQL detected. Statement type '{stmt_type}' is not allowed.")

    # Keyword-level scan as a safety net
    tokens = [token.value.upper() for token in parsed[0].flatten() if not token.is_whitespace]
    for token_val in tokens:
        if token_val in BLOCKED_KEYWORDS:
            raise ValueError(f"SECURITY ALERT: Harmful keyword '{token_val}' detected.")
            
def agent_workflow(user_question, chat_history=None):
    """
    Self-Healing Loop + Memory + Simulation + Network Fixes + Caching
    """
    if chat_history is None:
        chat_history = []

    print(f"Thinking about: {user_question}")
    
    cached_sql = get_cached_sql(user_question)
    if cached_sql:
        print("CACHE HIT: Skipping LLM generation.")
        data, error = run_query(cached_sql)
        if not error:
            return data, cached_sql
        else:
            print("   (Cache invalid, retrying with LLM...)")

    try:
        examples = get_few_shot_examples(user_question)
    except Exception as e:
        print(f"Retriever skipped: {e}")
        examples = ""

    schema = get_schema()

    system_prompt = f"""
    You are an expert SQL Data Analyst.
    You are querying a TPC-H database with CUSTOM column names.
    You have access to the following tables:
    {schema}

    RULES:
    1. Use ONLY DuckDB syntax.
    2. IMPORTANT: The schema has been modified.
       - Revenue is calculated as: sum(total_value * (1 - promo_reduction))
       - Do NOT use 'l_extendedprice' or 'l_discount'. Use 'total_value' and 'promo_reduction'.
    3. Return ONLY the SQL query. No markdown formatting. No explanation.
    4. If the user asks to delete or change data, politely refuse.

    5. COMPLEX QUERY HANDLING (CHAIN OF THOUGHT):
       - If a user asks a complex question (e.g. "Find the region with lowest revenue"), 
         you MUST use a CTE.
       - Example Logic: WITH regional_revenue AS (...) SELECT ...

    6. VISUALIZATION RULES (CRITICAL):
       - When grouping by a category, YOU MUST SELECT THE NAME, NOT THE ID.
       - CORRECT: SELECT r_name, sum(revenue)...
       
    7. COLUMN SECURITY:
       - If the user asks for a missing column (e.g. Profit), REFUSE and explain why.
       - Return text starting with "MISSING DATA:"

    Here are some examples:
    {examples}
    """

    if "what if" in user_question.lower() or "simulate" in user_question.lower():
        print("DETECTED SIMULATION INTENT")
        system_prompt += """
        \n\n SIMULATION MODE ACTIVE:
        Use a CTE to modify data temporarily (e.g., increase price by 10%).
        Then compare Original vs Simulated metrics.
        """
    
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_question))
    
    for attempt in range(3):
        print(f"  🔄 Attempt {attempt + 1}...")
        
        try:
            response = llm.invoke(messages)
            content = response.content.strip()
        except Exception as e:
            return None, f"NETWORK ERROR: {e}"
        
        if content.startswith("MISSING DATA") or content.startswith("I cannot"):
            return None, content 

        sql_query = content.replace("```sql", "").replace("```", "").strip()
        print(f"    -> Generated SQL: {sql_query}")
        
        try:
            check_sql_safety(sql_query)
        except ValueError as e:
            return None, f"I cannot execute that query. {e}"

        data, error = run_query(sql_query)
        
        if error:
            print(f"Error: {error}")
            messages.append(AIMessage(content=sql_query))
            messages.append(HumanMessage(content=f"That query failed with error: {error}. Please fix the SQL."))
        else:
            print("Success!")
            
            if "what if" not in user_question.lower():
                save_to_cache(user_question, sql_query)
                
            return data, sql_query

    return None, "Failed to generate valid SQL after 3 attempts."

if __name__ == "__main__":
    q = "What is the total revenue for the AFRICAN region?"
    result, final_sql = agent_workflow(q)
    print("\n--- FINAL RESULT ---")
    print(result)