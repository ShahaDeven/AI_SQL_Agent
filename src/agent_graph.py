import os
import sys
import json
import time
import warnings
import logging

# ===== SUPPRESS ALL WARNINGS/TELEMETRY BEFORE OTHER IMPORTS =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["CHROMA_TELEMETRY"] = "False"

# Suppress logging from noisy libraries
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry.product').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*telemetry.*')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import duckdb
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.retriever import get_few_shot_examples
from src.metrics import MetricsTracker
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
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

CACHE_FILE = os.path.join(PROJECT_ROOT, "sql_cache.json")

# Global metrics tracker (shared with app.py via import)
metrics = MetricsTracker()

def get_schema():
    """
    Returns the database schema to the LLM so it knows table names.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    schema_str = "Database Schema (DuckDB):\n"
    target_tables = ['customer', 'lineitem', 'orders', 'supplier', 'nation', 'part', 'region', 'partsupp']
    
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
            
        df = con.execute(clean_sql).fetchdf()  # FIX: Use clean_sql, not sql_query
        con.close()
        return df, None
    except Exception as e:
        return None, str(e)

def check_sql_safety(sql_query):
    """
    Parses SQL to ensure only READ-ONLY statements are executed.
    Blocks DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE, etc.
    """
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
    Self-Healing Loop + Memory + Simulation + Network Fixes + Caching + Metrics
    """
    if chat_history is None:
        chat_history = []

    # Start tracking this query
    metrics.begin_query(user_question)

    print(f"Thinking about: {user_question}")
    
    # --- Cache Check ---
    cached_sql = get_cached_sql(user_question)
    if cached_sql:
        print("CACHE HIT: Skipping LLM generation.")
        metrics.record_cache_hit()

        db_start = time.time()
        data, error = run_query(cached_sql)
        db_elapsed = time.time() - db_start
        
        if not error:
            metrics.record_db_execution(db_elapsed, rows=len(data))
            metrics.record_success(cached_sql)
            metrics.end_query()
            return data, cached_sql
        else:
            # FIX 1: Cache is invalid - log it and continue to LLM path
            print(f"   (Cache invalid: {error}, retrying with LLM...)")
            # Don't end_query here - let it continue to LLM generation

    # --- Retriever ---
    try:
        metrics.start_timer("retriever")
        examples = get_few_shot_examples(user_question)
        ret_elapsed = metrics.stop_timer("retriever")
        metrics.record_retriever_time(ret_elapsed)
    except Exception as e:
        metrics.stop_timer("retriever")
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

    is_simulation = ("what if" in user_question.lower() or 
                     "simulate" in user_question.lower() or 
                     "sensitivity" in user_question.lower() or
                     "scenario" in user_question.lower())

    if is_simulation:
        print("DETECTED SIMULATION INTENT")
        metrics.record_simulation()
        system_prompt += """

        SIMULATION MODE ACTIVE — FOLLOW THESE RULES STRICTLY:

        1. MULTI-VARIABLE SUPPORT:
           - The user may change MULTIPLE variables at once (e.g., "increase price by 5% AND reduce discount by 3%").
           - Apply ALL modifications in the same CTE.
           - If the user specifies a SCOPE (e.g., "for EUROPE only" or "for AUTOMOBILE segment"), 
             apply the modification ONLY to matching rows using CASE WHEN. Keep other rows unchanged.

        2. MANDATORY OUTPUT FORMAT — SIDE-BY-SIDE COMPARISON:
           - You MUST always return BOTH original AND simulated values in the same result set.
           - Required output columns: group column (if grouped), original_value, simulated_value, difference, pct_change.
           - If grouped (e.g., by region), show original vs simulated PER GROUP.
           - Always include: ROUND(((simulated - original) / original) * 100, 2) as pct_change
           - Example:
             WITH original_data AS (...), simulated_data AS (...)
             SELECT group_col, original_value, simulated_value,
                    (simulated_value - original_value) as difference,
                    ROUND(((simulated_value - original_value) / original_value) * 100, 2) as pct_change
             FROM original_data JOIN simulated_data ...

        3. SENSITIVITY ANALYSIS:
           - If the user asks "how sensitive is revenue to discount" or similar,
             generate a query that tests MULTIPLE levels (e.g., +5%, +10%, +15%, +20%).
           - Use UNION ALL to combine results from each level.
           - Output columns: scenario_label, original_value, simulated_value, difference, pct_change.
           - Example: 'Discount +5%', 'Discount +10%', etc. as scenario_label.

        4. ALWAYS use revenue formula: SUM(total_value * (1 - promo_reduction))
        """
    
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_question))
    
    prompt_chars = len(system_prompt) + len(user_question)
    
    for attempt in range(3):
        print(f"  🔄 Attempt {attempt + 1}...")
        
        try:
            llm_start = time.time()
            response = llm.invoke(messages)
            llm_elapsed = time.time() - llm_start
            content = response.content.strip()
            
            metrics.record_llm_call(
                elapsed=llm_elapsed,
                attempt=attempt + 1,
                prompt_chars=prompt_chars,
                completion_chars=len(content)
            )
        except Exception as e:
            metrics.record_error(f"NETWORK ERROR: {e}")
            metrics.end_query()
            return None, f"NETWORK ERROR: {e}"
        
        if content.startswith("MISSING DATA") or content.startswith("I cannot"):
            metrics.record_safety_refusal()
            metrics.record_error(content)
            metrics.end_query()
            return None, content 

        sql_query = content.replace("```sql", "").replace("```", "").strip()
        print(f"    -> Generated SQL: {sql_query}")
        
        try:
            check_sql_safety(sql_query)
        except ValueError as e:
            metrics.record_safety_refusal()
            metrics.record_error(str(e))
            metrics.end_query()
            return None, f"I cannot execute that query. {e}"

        db_start = time.time()
        data, error = run_query(sql_query)
        db_elapsed = time.time() - db_start
        
        if error:
            print(f"Error: {error}")
            messages.append(AIMessage(content=sql_query))
            messages.append(HumanMessage(content=f"That query failed with error: {error}. Please fix the SQL."))
            prompt_chars += len(sql_query) + len(error) + 50
        else:
            print("Success!")
            metrics.record_db_execution(db_elapsed, rows=len(data))
            metrics.record_success(sql_query)
            metrics.end_query()
            
            if not is_simulation:
                save_to_cache(user_question, sql_query)
                
            return data, sql_query

    metrics.record_error("Failed to generate valid SQL after 3 attempts.")
    metrics.end_query()
    return None, "Failed to generate valid SQL after 3 attempts."

if __name__ == "__main__":
    q = "What is the total revenue for the AFRICAN region?"
    result, final_sql = agent_workflow(q)
    print("\n--- FINAL RESULT ---")
    print(result)
    print("\n--- METRICS ---")
    print(json.dumps(metrics.get_summary(), indent=2))