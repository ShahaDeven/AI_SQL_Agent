import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import duckdb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retriever import get_few_shot_examples
import sqlparse

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
    raise FileNotFoundError(f"🚨 Critical Error: No database found! Checked: \n1. {DEMO_DB_PATH}\n2. {FULL_DB_PATH}")

MODEL_NAME = "gemini-2.5-flash" 
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

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

def run_query(sql_query):
    """
    Executes SQL. Returns (Result, Error).
    """
    try:
        clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()
        
        con = duckdb.connect(DB_PATH, read_only=True)
        if "DROP" in clean_sql.upper() or "DELETE" in clean_sql.upper():
            return None, "Security Error: Read-Only Access."
            
        # result = con.execute(clean_sql).fetchall()
        df = con.execute(sql_query).fetchdf()
        con.close()
        return df, None
    except Exception as e:
        return None, str(e)

def check_sql_safety(sql_query):
    """
    Parses SQL to ensure only READ-ONLY statements are executed.
    Blocks DROP, DELETE, INSERT, UPDATE, ALTER, etc.
    """
    # Parse the SQL
    parsed = sqlparse.parse(sql_query)

    for statement in parsed:
        stmt_type = statement.get_type().upper()

        if stmt_type not in ['SELECT', 'UNKNOWN']:
            raise ValueError(f"🚨 SECURITY ALERT: Harmful SQL detected. Statement type '{stmt_type}' is not allowed.")

    for token in parsed[0].flatten():
        if token.ttype is sqlparse.tokens.DML or token.ttype is sqlparse.tokens.DDL:
            if token.value.upper() in ['DELETE', 'UPDATE', 'DROP', 'INSERT', 'ALTER', 'TRUNCATE']:
                raise ValueError(f"🚨 SECURITY ALERT: Harmful keyword '{token.value.upper()}' detected.")
            
def agent_workflow(user_question):
    """
    The Self-Healing Loop.
    """
    print(f"Thinking about: {user_question}")
    
    # 1. Retrieve Examples
    examples = get_few_shot_examples(user_question)
    schema = get_schema()

    # 2. Dynamic System Prompt
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
       - If a user asks a complex question (e.g. "Find the region with lowest revenue and list its top supplier"), 
         you MUST use a CTE (Common Table Expression) to solve it in ONE step.
       - Example Logic:
         WITH regional_revenue AS (
             SELECT n_regionkey, SUM(...) as rev FROM ... GROUP BY n_regionkey ORDER BY rev ASC LIMIT 1
         )
         SELECT s_name FROM supplier JOIN nation ... WHERE n_regionkey = (SELECT n_regionkey FROM regional_revenue);

    6. VISUALIZATION RULES (CRITICAL):
       - When grouping by a category (Region, Nation, Customer), YOU MUST SELECT THE NAME, NOT THE ID.
       - WRONG: SELECT n_regionkey, sum(revenue)...
       - CORRECT: SELECT r_name, sum(revenue)...
       - Always JOIN the necessary tables (like 'region') to get the human-readable names.
       
    VISUALIZATION LOGIC:
    - If the user asks for a trend over time (years, months), suggest a LINE chart.
    - If the user compares categories (nations, regions, segments), suggest a BAR chart.
    - If the user asks for parts of a whole, suggest a PIE chart.
    - Otherwise, default to TABLE.
    
    NOTE: You cannot draw charts yourself. Just write the SQL to fetch the data. 
    The frontend will handle the rest based on the data shape.
    
    Here are some examples of how to write queries for this specific database:
    {examples}
    """
    
    # 🎯 PHASE 3: SIMULATION LOGIC INJECTION
    if "what if" in user_question.lower() or "simulate" in user_question.lower() or "impact" in user_question.lower():
        print("DETECTED SIMULATION INTENT")
        system_prompt += """
        \n\nSIMULATION MODE ACTIVE:
        The user is asking a "What If" question. You must use a CTE to modify the data temporarily.
        
        INSTRUCTIONS:
        1. Identify the variable the user wants to change (e.g., "increase discount by 5%").
        2. Create a CTE called 'simulated_data' that mimics the table but modifies that specific column.
           - Example: If changing 'lineitem', SELECT *, l_discount + 0.05 as l_discount FROM lineitem.
        3. Run the user's requested analysis on 'simulated_data' instead of the real table.
        4. If comparing, you can select the 'Original' metric and the 'Simulated' metric in the final SELECT.
        
        Example SQL Structure:
        WITH simulated_lineitem AS (
            SELECT *, total_value * 1.10 as total_value FROM lineitem -- Simulating 10% price hike
        )
        SELECT 
           sum(l.total_value) as original_revenue,
           sum(s.total_value) as simulated_revenue
        FROM lineitem l, simulated_lineitem s
        -- (Ideally join properly or just calculate the simulated one if that's what they asked)
        """

    messages = [
        ("system", system_prompt),
        ("human", f"Question: {user_question}")
    ]
    
    for attempt in range(3):
        print(f"Attempt {attempt + 1}...")

        response = llm.invoke(messages)
        sql_query = response.content
        
        # CLEANUP: Remove Markdown
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        print(f"    -> Generated SQL: {sql_query}")
        
        try:
            check_sql_safety(sql_query)
        except ValueError as e:
            return f"I cannot execute that query. {e}", sql_query

        data, error = run_query(sql_query)
        
        if error:
            print(f" Error: {error}")
            messages.append(("assistant", sql_query))
            messages.append(("human", f"That query failed with error: {error}. Please fix the SQL based on the schema provided."))
        else:
            print("Success!")
            return data, sql_query

    return None, "Failed to generate valid SQL after 3 attempts."

if __name__ == "__main__":
    q = "What if we increased the discount by 10%? How would that affect total revenue?"
    result, final_sql = agent_workflow(q)
    print("\n--- FINAL RESULT ---")
    print(result)